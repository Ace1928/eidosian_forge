import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
@staticmethod
def _is_subcycle(cycle, cycles):
    super_cycle = deepcopy(cycle)
    super_cycle = super_cycle[:-1]
    super_cycle.extend(cycle)
    for found_cycle in cycles:
        if len(found_cycle) == len(cycle) and is_sublist(super_cycle, found_cycle):
            return True
    return False