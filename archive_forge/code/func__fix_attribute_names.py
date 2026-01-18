from .... import symbol
from .... import  module
from .... import  context
from .... import  ndarray as nd
from .... import  io
def _fix_attribute_names(attrs, change_map):
    """
    Change attribute names as per values in change_map dictionary.
    Parameters
    ----------
    :param attrs : dict Dict of operator attributes
    :param change_map : dict Dict of onnx attribute name to mxnet attribute names.

    Returns
    -------
    :return new_attr : dict Converted dict of operator attributes.
    """
    new_attr = {}
    for k in attrs.keys():
        if k in change_map:
            new_attr[change_map[k]] = attrs[k]
        else:
            new_attr[k] = attrs[k]
    return new_attr