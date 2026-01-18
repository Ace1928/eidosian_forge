from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import handlers
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.command_lib.util.concepts import presentation_specs
import six
def GetExampleArgString(self):
    """Returns a command line example arg string for the concept."""
    examples = []
    for spec_name in self._specs:
        info = self.GetInfo(spec_name)
        args = info.GetExampleArgList()
        if args:
            examples.extend(args)

    def _PositionalsFirst(arg):
        prefix = 'Z' if arg.startswith('--') else 'A'
        return prefix + arg
    return ' '.join(sorted(examples, key=_PositionalsFirst))