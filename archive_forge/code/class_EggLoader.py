from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
class EggLoader(_Loader):

    def __init__(self, spec):
        self.spec = spec

    def get_context(self, object_type, name=None, global_conf=None):
        if self.absolute_name(name):
            return loadcontext(object_type, name, global_conf=global_conf)
        entry_point, protocol, ep_name = self.find_egg_entry_point(object_type, name=name)
        return LoaderContext(entry_point, object_type, protocol, global_conf or {}, {}, self, distribution=importlib_metadata.distribution(self.spec), entry_point_name=ep_name)

    def find_egg_entry_point(self, object_type, name=None):
        """
        Returns the (entry_point, protocol) for with the given ``name``.
        """
        if name is None:
            name = 'main'
        dist = importlib_metadata.distribution(self.spec)
        possible = []
        for protocol_options in object_type.egg_protocols:
            for protocol in protocol_options:
                entry = find_entry_point(dist, protocol, name)
                if entry is not None:
                    possible.append((entry.load(), protocol, entry.name))
                    break
        if not possible:
            raise LookupError('Entry point %r not found in egg %r (protocols: %s; entry_points: %s)' % (name, self.spec, ', '.join(_flatten(object_type.egg_protocols)), ', '.join((str(entry) for prot in protocol_options for entry in [find_entry_point(dist, prot, name)] if entry))))
        if len(possible) > 1:
            raise LookupError('Ambiguous entry points for %r in egg %r (protocols: %s)' % (name, self.spec, ', '.join(_flatten(protocol_options))))
        return possible[0]