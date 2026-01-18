import sys
import json
from .symbols import *
from .symbols import Symbol
class JsonDiffSyntax(object):

    def emit_set_diff(self, a, b, s, added, removed):
        raise NotImplementedError()

    def emit_list_diff(self, a, b, s, inserted, changed, deleted):
        raise NotImplementedError()

    def emit_dict_diff(self, a, b, s, added, changed, removed):
        raise NotImplementedError()

    def emit_value_diff(self, a, b, s):
        raise NotImplementedError()

    def patch(self, a, d):
        raise NotImplementedError()

    def unpatch(self, a, d):
        raise NotImplementedError()