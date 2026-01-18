from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class SettingDefinition(Statement):

    def __init__(self, name, value, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.value = value

    def __str__(self):
        if self.value is True:
            return f'{self.name}'
        if isinstance(self.value, (tuple, list)):
            value = ' '.join((str(v) for v in self.value))
            return f'{self.name} {value}'
        return f'{self.name} {self.value}'