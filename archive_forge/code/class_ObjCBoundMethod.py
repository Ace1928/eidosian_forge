import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class ObjCBoundMethod:
    """This represents an Objective-C method (an IMP) which has been bound
    to some id which will be passed as the first parameter to the method."""

    def __init__(self, method, objc_id):
        """Initialize with a method and ObjCInstance or ObjCClass object."""
        self.method = method
        self.objc_id = objc_id

    def __repr__(self):
        return '<ObjCBoundMethod %s (%s)>' % (self.method.name, self.objc_id)

    def __call__(self, *args):
        """Call the method with the given arguments."""
        return self.method(self.objc_id, *args)