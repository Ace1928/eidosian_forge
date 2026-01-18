from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
class Module_six_moves_urllib(types.ModuleType):
    """Create a six.moves.urllib namespace that resembles the Python 3 namespace"""
    __path__ = []
    parse = _importer._get_module('moves.urllib_parse')
    error = _importer._get_module('moves.urllib_error')
    request = _importer._get_module('moves.urllib_request')
    response = _importer._get_module('moves.urllib_response')
    robotparser = _importer._get_module('moves.urllib_robotparser')

    def __dir__(self):
        return ['parse', 'error', 'request', 'response', 'robotparser']