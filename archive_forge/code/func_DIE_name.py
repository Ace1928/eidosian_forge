from ..common.utils import bytes2str
def DIE_name(die):
    return bytes2str(die.attributes['DW_AT_name'].value)