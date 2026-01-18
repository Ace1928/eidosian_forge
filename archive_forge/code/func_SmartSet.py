import atexit
import os
import sys
import tempfile
from absl import app
from absl.testing.absltest import *
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
def SmartSet(self, obj, attr_name, new_attr):
    """Replace obj.attr_name with new_attr.

    This method is smart and works at the module, class, and instance level
    while preserving proper inheritance. It will not stub out C types however
    unless that has been explicitly allowed by the type.

    This method supports the case where attr_name is a staticmethod or a
    classmethod of obj.

    Notes:
      - If obj is an instance, then it is its class that will actually be
        stubbed. Note that the method Set() does not do that: if obj is
        an instance, it (and not its class) will be stubbed.
      - The stubbing is using the builtin getattr and setattr. So, the __get__
        and __set__ will be called when stubbing (TODO: A better idea would
        probably be to manipulate obj.__dict__ instead of getattr() and
        setattr()).

    Args:
      obj: The object whose attributes we want to modify.
      attr_name: The name of the attribute to modify.
      new_attr: The new value for the attribute.

    Raises:
      AttributeError: If the attribute cannot be found.
    """
    _, obj = tf_decorator.unwrap(obj)
    if tf_inspect.ismodule(obj) or (not tf_inspect.isclass(obj) and attr_name in obj.__dict__):
        orig_obj = obj
        orig_attr = getattr(obj, attr_name)
    else:
        if not tf_inspect.isclass(obj):
            mro = list(tf_inspect.getmro(obj.__class__))
        else:
            mro = list(tf_inspect.getmro(obj))
        mro.reverse()
        orig_attr = None
        found_attr = False
        for cls in mro:
            try:
                orig_obj = cls
                orig_attr = getattr(obj, attr_name)
                found_attr = True
            except AttributeError:
                continue
        if not found_attr:
            raise AttributeError('Attribute not found.')
    old_attribute = obj.__dict__.get(attr_name)
    if old_attribute is not None and isinstance(old_attribute, staticmethod):
        orig_attr = staticmethod(orig_attr)
    self.stubs.append((orig_obj, attr_name, orig_attr))
    setattr(orig_obj, attr_name, new_attr)