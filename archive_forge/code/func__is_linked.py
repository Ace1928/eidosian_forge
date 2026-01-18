import logging
from collections import defaultdict, namedtuple
from copy import deepcopy
def _is_linked(self, src, dst, pairs, current_path):
    if src == dst:
        return False
    if not pairs:
        return False
    if ConsumerPair(src, dst) in pairs:
        current_path.append(src)
        current_path.append(dst)
        return True
    for pair in pairs:
        if pair.src_member_id == src:
            reduced_set = deepcopy(pairs)
            reduced_set.remove(pair)
            current_path.append(pair.src_member_id)
            return self._is_linked(pair.dst_member_id, dst, reduced_set, current_path)
    return False