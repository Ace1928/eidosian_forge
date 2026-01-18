import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
class VScale(orig_VScale):

    def __init__(self, adjustment=None):
        orig_VScale.__init__(self, adjustment=adjustment)