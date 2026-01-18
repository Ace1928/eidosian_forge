from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _remove_attributes(attrs, remove_list):
    """
    Removes attributes in the remove list from the input attribute dict
    :param attrs : Dict of operator attributes
    :param remove_list : list of attributes to be removed

    :return new_attr : Dict of operator attributes without the listed attributes.
    """
    new_attrs = {}
    for attr in attrs.keys():
        if attr not in remove_list:
            new_attrs[attr] = attrs[attr]
    return new_attrs