import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
class HScale(orig_HScale):

    def __init__(self, adjustment=None):
        orig_HScale.__init__(self, adjustment=adjustment)