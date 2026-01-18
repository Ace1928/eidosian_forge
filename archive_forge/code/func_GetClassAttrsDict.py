from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
import types
from fire import docstrings
import six
def GetClassAttrsDict(component):
    """Gets the attributes of the component class, as a dict with name keys."""
    if not inspect.isclass(component):
        return None
    class_attrs_list = inspect.classify_class_attrs(component)
    return {class_attr.name: class_attr for class_attr in class_attrs_list}