from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.infra_manager import configmanager_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.infra_manager import deploy_util
from googlecloudsdk.command_lib.infra_manager import flags
from googlecloudsdk.command_lib.infra_manager import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ImportStatefile(base.Command):
    """Import a terraform state file.

  This command generates a signed url to upload a terraform state file.
  """
    detailed_help = {'EXAMPLES': '\n        Import state file for `my-deployment` with lock-id `1658343229583347`:\n\n          $ {command} projects/p1/locations/us-central1/deployments/my-deployment --lock-id=1658343229583347\n\n      '}

    @staticmethod
    def Args(parser):
        file_help_text = '        File path for importing statefile into a deployment. It specifies the\n        local file path of an existing Terraform statefile to directly upload\n        for a deployment.'
        flags.AddFileFlag(parser, file_help_text)
        flags.AddLockFlag(parser)
        concept_parsers.ConceptParser([resource_args.GetDeploymentResourceArgSpec('the deployment to be used as parent.')]).AddToParser(parser)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      A statefile containing signed url that can be used to upload state file.
    """
        messages = configmanager_util.GetMessagesModule()
        deployment_ref = args.CONCEPTS.deployment.Parse()
        deployment_full_name = deployment_ref.RelativeName()
        return deploy_util.ImportStateFile(messages, deployment_full_name, args.lock_id, args.file)