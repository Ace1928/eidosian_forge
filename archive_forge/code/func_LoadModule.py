import platform
import types
import unittest
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import test_util
def LoadModule(self, module_name, source):
    result = {'__name__': module_name, 'messages': messages}
    exec(source, result)
    module = types.ModuleType(module_name)
    for name, value in result.items():
        setattr(module, name, value)
    return module