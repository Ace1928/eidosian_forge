import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
class BaseConfigurator(object):
    """
    The configurator base class which defines some useful defaults.
    """
    CONVERT_PATTERN = re.compile('^(?P<prefix>[a-z]+)://(?P<suffix>.*)$')
    WORD_PATTERN = re.compile('^\\s*(\\w+)\\s*')
    DOT_PATTERN = re.compile('^\\.\\s*(\\w+)\\s*')
    INDEX_PATTERN = re.compile('^\\[\\s*(\\w+)\\s*\\]\\s*')
    DIGIT_PATTERN = re.compile('^\\d+$')
    value_converters = {'ext': 'ext_convert', 'cfg': 'cfg_convert'}
    importer = staticmethod(__import__)

    def __init__(self, config):
        self.config = ConvertingDict(config)
        self.config.configurator = self

    def resolve(self, s):
        """
        Resolve strings to objects using standard import and attribute
        syntax.
        """
        name = s.split('.')
        used = name.pop(0)
        try:
            found = self.importer(used)
            for frag in name:
                used += '.' + frag
                try:
                    found = getattr(found, frag)
                except AttributeError:
                    self.importer(used)
                    found = getattr(found, frag)
            return found
        except ImportError as e:
            v = ValueError('Cannot resolve %r: %s' % (s, e))
            raise v from e

    def ext_convert(self, value):
        """Default converter for the ext:// protocol."""
        return self.resolve(value)

    def cfg_convert(self, value):
        """Default converter for the cfg:// protocol."""
        rest = value
        m = self.WORD_PATTERN.match(rest)
        if m is None:
            raise ValueError('Unable to convert %r' % value)
        else:
            rest = rest[m.end():]
            d = self.config[m.groups()[0]]
            while rest:
                m = self.DOT_PATTERN.match(rest)
                if m:
                    d = d[m.groups()[0]]
                else:
                    m = self.INDEX_PATTERN.match(rest)
                    if m:
                        idx = m.groups()[0]
                        if not self.DIGIT_PATTERN.match(idx):
                            d = d[idx]
                        else:
                            try:
                                n = int(idx)
                                d = d[n]
                            except TypeError:
                                d = d[idx]
                if m:
                    rest = rest[m.end():]
                else:
                    raise ValueError('Unable to convert %r at %r' % (value, rest))
        return d

    def convert(self, value):
        """
        Convert values to an appropriate type. dicts, lists and tuples are
        replaced by their converting alternatives. Strings are checked to
        see if they have a conversion format and are converted if they do.
        """
        if not isinstance(value, ConvertingDict) and isinstance(value, dict):
            value = ConvertingDict(value)
            value.configurator = self
        elif not isinstance(value, ConvertingList) and isinstance(value, list):
            value = ConvertingList(value)
            value.configurator = self
        elif not isinstance(value, ConvertingTuple) and isinstance(value, tuple) and (not hasattr(value, '_fields')):
            value = ConvertingTuple(value)
            value.configurator = self
        elif isinstance(value, str):
            m = self.CONVERT_PATTERN.match(value)
            if m:
                d = m.groupdict()
                prefix = d['prefix']
                converter = self.value_converters.get(prefix, None)
                if converter:
                    suffix = d['suffix']
                    converter = getattr(self, converter)
                    value = converter(suffix)
        return value

    def configure_custom(self, config):
        """Configure an object with a user-supplied factory."""
        c = config.pop('()')
        if not callable(c):
            c = self.resolve(c)
        kwargs = {k: config[k] for k in config if k != '.' and valid_ident(k)}
        result = c(**kwargs)
        props = config.pop('.', None)
        if props:
            for name, value in props.items():
                setattr(result, name, value)
        return result

    def as_tuple(self, value):
        """Utility function which converts lists to tuples."""
        if isinstance(value, list):
            value = tuple(value)
        return value