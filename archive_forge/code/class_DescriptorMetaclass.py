import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
class DescriptorMetaclass(type):

    def __instancecheck__(cls, obj):
        if super(DescriptorMetaclass, cls).__instancecheck__(obj):
            return True
        if isinstance(obj, cls._C_DESCRIPTOR_CLASS):
            return True
        return False