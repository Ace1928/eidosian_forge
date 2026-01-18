import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
def _make_value_freed_callback(object_ref, name):

    def _value_freed(value_ref):
        object = object_ref()
        if object is not None:
            object.trait_property_changed(name, Undefined, None)
    return _value_freed