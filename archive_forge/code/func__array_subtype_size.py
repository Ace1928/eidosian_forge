from ..common.utils import bytes2str
def _array_subtype_size(sub):
    if 'DW_AT_upper_bound' in sub.attributes:
        return sub.attributes['DW_AT_upper_bound'].value + 1
    if 'DW_AT_count' in sub.attributes:
        return sub.attributes['DW_AT_count'].value
    else:
        return -1