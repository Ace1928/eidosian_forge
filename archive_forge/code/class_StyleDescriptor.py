import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
class StyleDescriptor(object):

    def __get__(self, instance, class_):
        return Styles(instance)