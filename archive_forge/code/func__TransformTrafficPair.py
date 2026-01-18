from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _TransformTrafficPair(pair):
    """Transforms a single TrafficTargetPair into a marker class structure."""
    console = console_attr.GetConsoleAttr()
    return (pair.displayPercent, console.Emphasize(pair.displayRevisionId), cp.Table([('', _GetTagAndStatus(t), t.url) for t in pair.tags]))