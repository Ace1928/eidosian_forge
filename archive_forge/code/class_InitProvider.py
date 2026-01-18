from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.declarative import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class InitProvider(base.DeclarativeCommand):
    """Generate main.tf file to configure Google Cloud Terraform Provider."""
    detailed_help = _DETAILED_HELP

    def _GetBillingParams(self, args_namspace):
        """Process billing project flags in args and return Terraform settings."""
        use_gcloud_billing = args_namspace.use_gcloud_billing_project
        user_project_override = billing_project = None
        if use_gcloud_billing:
            billing_project = properties.VALUES.billing.quota_project.Get()
            user_project_override = 'true'
        elif args_namspace.tf_user_project_override:
            billing_project = args_namspace.tf_billing_project
            user_project_override = 'true'
        return (user_project_override, billing_project)

    @classmethod
    def Args(cls, parser):
        flags.AddInitProviderArgs(parser)

    def Run(self, args):
        do_override, billing_project = self._GetBillingParams(args)
        project = args.project or properties.VALUES.core.project.Get()
        region = args.region or properties.VALUES.compute.region.Get()
        zone = args.zone or properties.VALUES.compute.zone.Get()
        template_context = {'project': project, 'region': region, 'zone': zone, 'user_override': do_override, 'billing_project': billing_project}
        path = os.path.join(files.GetCWD(), 'main.tf')
        if os.path.isfile(path):
            console_io.PromptContinue('{} Exists.'.format(path), prompt_string='Overwrite?', cancel_on_no=True, cancel_string='Init Provider cancelled.')
        with progress_tracker.ProgressTracker('Creating Terraform init module'):
            with files.FileWriter(path, create_path=True) as f:
                ctx = runtime.Context(f, **template_context)
                INIT_FILE_TEMPLATE.render_context(ctx)
        log.status.Print('Created Terraform module file {path}.'.format(path=path))