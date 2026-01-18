from __future__ import (absolute_import, division, print_function)
class PrefixFactNamespace(FactNamespace):

    def __init__(self, namespace_name, prefix=None):
        super(PrefixFactNamespace, self).__init__(namespace_name)
        self.prefix = prefix

    def transform(self, name):
        new_name = self._underscore(name)
        return '%s%s' % (self.prefix, new_name)