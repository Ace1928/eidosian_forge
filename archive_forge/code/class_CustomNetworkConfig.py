from __future__ import absolute_import, division, print_function
import hashlib
import re
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.six.moves import zip
class CustomNetworkConfig(NetworkConfig):

    def items_text(self):
        return [item.text for item in self.items]

    def expand_section(self, configobj, S=None):
        if S is None:
            S = list()
        S.append(configobj)
        for child in configobj.child_objs:
            if child in S:
                continue
            self.expand_section(child, S)
        return S

    def to_block(self, section):
        return '\n'.join([item.raw for item in section])

    def get_section(self, path):
        try:
            section = self.get_section_objects(path)
            return self.to_block(section)
        except ValueError:
            return list()

    def get_section_objects(self, path):
        if not isinstance(path, list):
            path = [path]
        obj = self.get_object(path)
        if not obj:
            raise ValueError('path does not exist in config')
        return self.expand_section(obj)