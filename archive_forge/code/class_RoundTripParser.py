from __future__ import absolute_import
from ruamel.yaml.error import MarkedYAMLError
from ruamel.yaml.tokens import *  # NOQA
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.scanner import Scanner, RoundTripScanner, ScannerError  # NOQA
from ruamel.yaml.compat import utf8, nprint, nprintf  # NOQA
class RoundTripParser(Parser):
    """roundtrip is a safe loader, that wants to see the unmangled tag"""

    def transform_tag(self, handle, suffix):
        if handle == '!!' and suffix in (u'null', u'bool', u'int', u'float', u'binary', u'timestamp', u'omap', u'pairs', u'set', u'str', u'seq', u'map'):
            return Parser.transform_tag(self, handle, suffix)
        return handle + suffix