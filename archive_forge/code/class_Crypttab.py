from __future__ import absolute_import, division, print_function
import os
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_native
class Crypttab(object):
    _lines = []

    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            open(path, 'a').close()
        try:
            f = open(path, 'r')
            for line in f.readlines():
                self._lines.append(Line(line))
        finally:
            f.close()

    def add(self, line):
        self._lines.append(line)
        return (True, 'added line')

    def lines(self):
        for line in self._lines:
            if line.valid():
                yield line

    def match(self, name):
        for line in self.lines():
            if line.name == name:
                return line
        return None

    def __str__(self):
        lines = []
        for line in self._lines:
            lines.append(str(line))
        crypttab = '\n'.join(lines)
        if len(crypttab) == 0:
            crypttab += '\n'
        if crypttab[-1] != '\n':
            crypttab += '\n'
        return crypttab