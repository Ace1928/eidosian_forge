from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six
def ParseNodeFlag(node_flag, node_specs):
    """Parses the --node flag into a list of node_specs."""
    num_nodes = len(node_specs)
    if six.text_type(node_flag).upper() == 'ALL':
        indexes = list(range(num_nodes))
    else:
        indexes = set()
        ranges = node_flag.split(',')
        for r in ranges:
            if not r:
                continue
            if '-' in r:
                bounds = r.split('-')
                if len(bounds) != 2 or not bounds[0] or (not bounds[1]):
                    raise exceptions.InvalidArgumentException('--node', 'Range "{}" does not match expected format "lowerBound-upperBound", where lowerBound < upperBound.'.format(r))
                start, end = (int(bounds[0]), int(bounds[1]))
                if start >= end:
                    raise exceptions.InvalidArgumentException('--node', 'Range "{}" does not match expected format "lowerBound-upperBound", where lowerBound < upperBound.'.format(r))
                indexes.update(range(start, end + 1))
            else:
                try:
                    indexes.add(int(r))
                except ValueError:
                    raise exceptions.InvalidArgumentException('--node', 'unable to parse node ID {}. Please only use numbers.'.format(r))
    if not indexes:
        raise exceptions.InvalidArgumentException('--node', 'Unable to parse node ranges from {}.'.format(node_flag))
    mx = max(indexes)
    if mx >= num_nodes:
        raise exceptions.InvalidArgumentException('--node', 'node index {} is larger than the valid node indices on this TPU Queued Resource. Please only use indexes in the range [0, {}], inclusive.'.format(mx, num_nodes - 1))
    filtered_node_specs = []
    for node in indexes:
        filtered_node_specs.append(node_specs[node])
    return filtered_node_specs