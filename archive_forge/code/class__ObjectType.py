from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class _ObjectType:
    name = None
    egg_protocols = None
    config_prefixes = None

    def __init__(self):
        self.egg_protocols = [_aslist(p) for p in _aslist(self.egg_protocols)]
        self.config_prefixes = [_aslist(p) for p in _aslist(self.config_prefixes)]

    def __repr__(self):
        return '<{} protocols={!r} prefixes={!r}>'.format(self.name, self.egg_protocols, self.config_prefixes)

    def invoke(self, context):
        assert context.protocol in _flatten(self.egg_protocols)
        return fix_call(context.object, context.global_conf, **context.local_conf)