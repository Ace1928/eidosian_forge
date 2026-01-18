from googlecloudsdk.api_lib.storage import insights_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage.insights.dataset_configs import log_util
from googlecloudsdk.command_lib.storage.insights.dataset_configs import resource_args
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class DeleteLink(base.Command):
    """Delete a link to a BigQuery instance."""
    detailed_help = {'DESCRIPTION': '\n      Delete a link to a BigQuery instance.\n      ', 'EXAMPLES': '\n\n      To unlink a dataset config with config name "my-config" in location\n      "us-central1":\n\n          $ {command} my-config --location=us-central1\n\n      To delete a link for the same dataset config with fully specified name:\n\n          $ {command} projects/foo/locations/us-central1/datasetConfigs/my-config\n      '}

    @staticmethod
    def Args(parser):
        resource_args.add_dataset_config_resource_arg(parser, 'to delete link')

    def Run(self, args):
        client = insights_api.InsightsApi()
        dataset_config_relative_name = args.CONCEPTS.dataset_config.Parse().RelativeName()
        delete_dataset_config_link_operation = client.delete_dataset_config_link(dataset_config_relative_name)
        log_util.dataset_config_operation_started_and_status_log('Delete link', dataset_config_relative_name, delete_dataset_config_link_operation.name)