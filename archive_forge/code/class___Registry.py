import copy
import re
class __Registry:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state
        self.__context = {}

    def __getitem__(self, string):
        try:
            return eval(string, self.__context)
        except NameError:
            raise LookupError('Unable to parse units: "%s"' % string)

    def __setitem__(self, string, val):
        assert isinstance(string, str)
        try:
            assert string not in self.__context
        except AssertionError:
            if val == self.__context[string]:
                return
            raise KeyError('%s has already been registered for %s' % (string, self.__context[string]))
        self.__context[string] = val