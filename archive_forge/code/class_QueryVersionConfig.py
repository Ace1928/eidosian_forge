from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkeonprem import vmware_clusters as apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.command_lib.container.vmware import flags
from googlecloudsdk.core import log
import six
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA, base.ReleaseTrack.GA)
class QueryVersionConfig(base.Command):
    """Query versions for creating or upgrading an Anthos on VMware user cluster."""
    detailed_help = {'EXAMPLES': _EXAMPLES}

    @staticmethod
    def Args(parser: parser_arguments.ArgumentInterceptor):
        """Registers flags for this command."""
        flags.AddLocationResourceArg(parser, 'to query versions')
        flags.AddConfigType(parser)

    def Run(self, args):
        """Runs the query-version-config command."""
        client = apis.ClustersClient()
        return client.QueryVersionConfig(args)

    def Epilog(self, resources_were_displayed):
        super(QueryVersionConfig, self).Epilog(resources_were_displayed)
        command_base = 'gcloud'
        if self.ReleaseTrack() is base.ReleaseTrack.BETA or self.ReleaseTrack() is base.ReleaseTrack.ALPHA:
            command_base += ' ' + six.text_type(self.ReleaseTrack()).lower()
        log.status.Print(_EPILOG.format(command_base))