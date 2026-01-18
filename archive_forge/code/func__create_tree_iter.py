import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def _create_tree_iter(self, data):
    """Internal creation of a (bool, TreeIter) pair for returning directly
        back to the view interfacing with this model."""
    if data is None:
        return (False, None)
    else:
        it = self.create_tree_iter(data)
        return (True, it)