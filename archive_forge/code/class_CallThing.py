import os
from collections import abc
from functools import partial
from gi.repository import GLib, GObject, Gio
class CallThing(object):

    def __init__(self, name, func):
        self._name = name
        self._func = func