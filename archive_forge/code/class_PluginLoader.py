from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
class PluginLoader:

    def __init__(self, group):
        self.group = group
        self.impls = {}

    def load(self, name):
        if name in self.impls:
            return self.impls[name]()
        for impl in importlib_metadata_get(self.group):
            if impl.name == name:
                self.impls[name] = impl.load
                return impl.load()
        from mako import exceptions
        raise exceptions.RuntimeException("Can't load plugin %s %s" % (self.group, name))

    def register(self, name, modulepath, objname):

        def load():
            mod = __import__(modulepath)
            for token in modulepath.split('.')[1:]:
                mod = getattr(mod, token)
            return getattr(mod, objname)
        self.impls[name] = load