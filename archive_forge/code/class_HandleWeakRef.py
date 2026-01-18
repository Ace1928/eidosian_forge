import enum
import os
import sys
from os import getcwd
from os.path import dirname, exists, join
from weakref import ref
from .etsconfig.api import ETSConfig
class HandleWeakRef(object):

    def __init__(self, object, name, value):
        object_ref = ref(object)
        _value_freed = _make_value_freed_callback(object_ref, name)
        self.object = object_ref
        self.name = name
        self.value = ref(value, _value_freed)