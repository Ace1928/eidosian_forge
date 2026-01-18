from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.code import kubernetes
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class CleanUp(base.Command):
    """Delete the local development environment."""
    detailed_help = {'DESCRIPTION': '          Delete the local development environment.\n\n          Use this command to clean up a development environment. This command\n          many also be used remove any artifacts of developments environments\n          that did not successfully start up.\n          ', 'EXAMPLES': '          $ {command}\n\n          To clean up a specific profile:\n\n          $ {command} --minikube-profile=<profile name>\n          '}

    @classmethod
    def Args(cls, parser):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--minikube-profile', help='Minikube profile.')

    def Run(self, args):
        kubernetes.DeleteMinikube(args.minikube_profile or kubernetes.DEFAULT_CLUSTER_NAME)