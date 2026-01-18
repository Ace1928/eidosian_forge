from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.core.updater import local_state
@base.Hidden
class PostProcess(base.SilentCommand):
    """Performs any necessary post installation steps."""

    @staticmethod
    def Args(parser):
        parser.add_argument('--force-recompile', action='store_true', required=False, hidden=True, default='False', help='THIS ARGUMENT NEEDS HELP TEXT.')
        parser.add_argument('--compile-python', required=False, hidden=True, default='True', action='store_true', help='THIS ARGUMENT NEEDS HELP TEXT.')

    def Run(self, args):
        if args.compile_python:
            state = local_state.InstallationState.ForCurrent()
            state.CompilePythonFiles(force=args.force_recompile)