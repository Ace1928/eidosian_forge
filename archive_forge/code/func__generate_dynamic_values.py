from collections import namedtuple
from io import BytesIO
from ..common.utils import struct_parse, bytelist2string, read_blob
from ..common.exceptions import DWARFError
def _generate_dynamic_values(map, prefix, index_start, index_end, value_start):
    """ Generate values in a map (dict) dynamically. Each key starts with
        a (string) prefix, followed by an index in the inclusive range
        [index_start, index_end]. The values start at value_start.
    """
    for index in range(index_start, index_end + 1):
        name = '%s%s' % (prefix, index)
        value = value_start + index - index_start
        map[name] = value