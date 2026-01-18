from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class LangSysDefinition(Statement):

    def __init__(self, name, tag, features, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.tag = tag
        self.features = features

    def __str__(self):
        res = 'DEF_LANGSYS'
        if self.name is not None:
            res += f' NAME "{self.name}"'
        res += f' TAG "{self.tag}"\n\n'
        for feature in self.features:
            res += f'{feature}'
        res += 'END_LANGSYS\n'
        return res