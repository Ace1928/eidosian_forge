from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _TransformInstanceSplitPairs(instance_split_pairs):
    """Transforms a List[TrafficTargetPair] into a marker class structure."""
    instance_split_section = cp.Section([cp.Table((_TransformInstanceSplitPair(p) for p in instance_split_pairs))])
    return cp.Section([cp.Labeled([('Instance Split', instance_split_section)])], max_column_width=60)