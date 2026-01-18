from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os.path
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.credentials.store import GetFreshAccessToken
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class Vet(base.Command):
    """Validate that a terraform plan complies with policies."""
    detailed_help = {'EXAMPLES': '\n        To validate that a terraform plan complies with a policy library\n        at `/my/policy/library`:\n\n        $ {command} tfplan.json --policy-library=/my/policy/library\n        '}

    @staticmethod
    def Args(parser):
        parser.add_argument('terraform_plan_json', help='File which contains a JSON export of a terraform plan. This file will be validated against the given policy library.')
        parser.add_argument('--policy-library', required=True, help='Directory which contains a policy library')
        parser.add_argument('--zone', required=False, help='Default zone to use for resources that do not have one set')
        parser.add_argument('--region', required=False, help='Default region to use for resources that do not have one set')

    def Run(self, args):
        tfplan_to_cai_operation = TerraformToolsTfplanToCaiOperation()
        validate_cai_operation = TerraformToolsValidateOperation()
        validate_tfplan_operation = TerraformToolsValidateOperation()
        env_vars = {'GOOGLE_OAUTH_ACCESS_TOKEN': GetFreshAccessToken(account=properties.VALUES.core.account.Get()), 'USE_STRUCTURED_LOGGING': 'true'}
        proxy_env_names = ['HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'NO_PROXY', 'no_proxy']
        project_env_names = ['GOOGLE_PROJECT', 'GOOGLE_CLOUD_PROJECT', 'GCLOUD_PROJECT']
        zone_env_names = ['GOOGLE_ZONE', 'GCLOUD_ZONE', 'CLOUDSDK_COMPUTE_ZONE']
        region_env_names = ['GOOGLE_REGION', 'GCLOUD_REGION', 'CLOUDSDK_COMPUTE_REGION']
        for env_key, env_val in os.environ.items():
            if env_key in proxy_env_names:
                env_vars[env_key] = env_val
        with files.TemporaryDirectory() as tempdir:
            cai_assets = os.path.join(tempdir, 'cai_assets.json')
            project = properties.VALUES.core.project.Get()
            if project:
                log.debug('Setting project to {} from properties'.format(project))
            else:
                for env_key in project_env_names:
                    project = encoding.GetEncodedValue(os.environ, env_key)
                    if project:
                        log.debug('Setting project to {} from env {}'.format(project, env_key))
                        break
            region = ''
            if args.region:
                region = args.region
                log.debug('Setting region to {} from args'.format(region))
            else:
                for env_key in region_env_names:
                    region = encoding.GetEncodedValue(os.environ, env_key)
                    if region:
                        log.debug('Setting region to {} from env {}'.format(region, env_key))
                        break
            zone = ''
            if args.zone:
                zone = args.zone
                log.debug('Setting zone to {} from args'.format(zone))
            else:
                for env_key in zone_env_names:
                    zone = encoding.GetEncodedValue(os.environ, env_key)
                    if zone:
                        log.debug('Setting zone to {} from env {}'.format(zone, env_key))
                        break
            response = tfplan_to_cai_operation(command='tfplan-to-cai', project=project, region=region, zone=zone, terraform_plan_json=args.terraform_plan_json, verbosity=args.verbosity, output_path=cai_assets, env=env_vars)
            self.exit_code = response.exit_code
            if self.exit_code > 0:
                return None
            with progress_tracker.ProgressTracker(message='Validating resources', aborted_message='Aborted validation.'):
                cai_response = validate_cai_operation(command='validate-cai', policy_library=args.policy_library, input_file=cai_assets, verbosity=args.verbosity, env=env_vars)
                tfplan_response = validate_tfplan_operation(command='validate-tfplan', policy_library=args.policy_library, input_file=args.terraform_plan_json, verbosity=args.verbosity, env=env_vars)
        if cai_response.exit_code == 1 or tfplan_response.exit_code == 1:
            self.exit_code = 1
        elif cai_response.exit_code == 2 or tfplan_response.exit_code == 2:
            self.exit_code = 2
        violations = []
        for policy_type, response in (('CAI', cai_response), ('Terraform', tfplan_response)):
            if response.stdout:
                try:
                    msg = binary_operations.ReadStructuredOutput(response.stdout, as_json=True)
                except binary_operations.StructuredOutputError:
                    log.warning('Could not parse {} policy validation output.'.format(policy_type))
                else:
                    violations += msg.resource_body
            if response.stderr:
                handler = binary_operations.DefaultStreamStructuredErrHandler(None)
                for line in response.stderr.split('\n'):
                    handler(line)
        return violations