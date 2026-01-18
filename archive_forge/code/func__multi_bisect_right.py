from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
@staticmethod
def _multi_bisect_right(in_keys, fixed_keys):
    """Find the positions where each 'in_key' would fit in fixed_keys.

        This is equivalent to doing "bisect_right" on each in_key into
        fixed_keys

        :param in_keys: A sorted list of keys to match with fixed_keys
        :param fixed_keys: A sorted list of keys to match against
        :return: A list of (integer position, [key list]) tuples.
        """
    if not in_keys:
        return []
    if not fixed_keys:
        return [(0, in_keys)]
    if len(in_keys) == 1:
        return [(bisect.bisect_right(fixed_keys, in_keys[0]), in_keys)]
    in_keys_iter = iter(in_keys)
    fixed_keys_iter = enumerate(fixed_keys)
    cur_in_key = next(in_keys_iter)
    cur_fixed_offset, cur_fixed_key = next(fixed_keys_iter)

    class InputDone(Exception):
        pass

    class FixedDone(Exception):
        pass
    output = []
    cur_out = []
    try:
        while True:
            if cur_in_key < cur_fixed_key:
                cur_keys = []
                cur_out = (cur_fixed_offset, cur_keys)
                output.append(cur_out)
                while cur_in_key < cur_fixed_key:
                    cur_keys.append(cur_in_key)
                    try:
                        cur_in_key = next(in_keys_iter)
                    except StopIteration as exc:
                        raise InputDone from exc
            while cur_in_key >= cur_fixed_key:
                try:
                    cur_fixed_offset, cur_fixed_key = next(fixed_keys_iter)
                except StopIteration as exc:
                    raise FixedDone from exc
    except InputDone:
        pass
    except FixedDone:
        cur_keys = [cur_in_key]
        cur_keys.extend(in_keys_iter)
        cur_out = (len(fixed_keys), cur_keys)
        output.append(cur_out)
    return output