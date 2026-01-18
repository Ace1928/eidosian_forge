from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _TransformTrafficPairs(traffic_pairs, service_url, service_ingress=None):
    """Transforms a List[TrafficTargetPair] into a marker class structure."""
    traffic_section = cp.Section([cp.Table((_TransformTrafficPair(p) for p in traffic_pairs))])
    route_section = [cp.Labeled([('URL', service_url)])]
    if service_ingress is not None:
        route_section.append(cp.Labeled([('Ingress', service_ingress)]))
    route_section.append(cp.Labeled([('Traffic', traffic_section)]))
    return cp.Section(route_section, max_column_width=60)