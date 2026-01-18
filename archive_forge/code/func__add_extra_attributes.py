from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _add_extra_attributes(attrs, extra_attr_map):
    """
    :param attrs:  Current Attribute list
    :param extraAttrMap:  Additional attributes to be added
    :return: new_attr
    """
    for attr in extra_attr_map:
        if attr not in attrs:
            attrs[attr] = extra_attr_map[attr]
    return attrs