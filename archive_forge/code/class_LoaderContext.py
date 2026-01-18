from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class LoaderContext:

    def __init__(self, obj, object_type, protocol, global_conf, local_conf, loader, distribution=None, entry_point_name=None):
        self.object = obj
        self.object_type = object_type
        self.protocol = protocol
        self.global_conf = global_conf
        self.local_conf = local_conf
        self.loader = loader
        self.distribution = distribution
        self.entry_point_name = entry_point_name

    def create(self):
        return self.object_type.invoke(self)

    def config(self):
        conf = AttrDict(self.global_conf)
        conf.update(self.local_conf)
        conf.local_conf = self.local_conf
        conf.global_conf = self.global_conf
        conf.context = self
        return conf