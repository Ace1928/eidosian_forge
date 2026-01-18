from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class ScriptDefinition(Statement):

    def __init__(self, name, tag, langs, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.tag = tag
        self.langs = langs

    def __str__(self):
        res = 'DEF_SCRIPT'
        if self.name is not None:
            res += f' NAME "{self.name}"'
        res += f' TAG "{self.tag}"\n\n'
        for lang in self.langs:
            res += f'{lang}'
        res += 'END_SCRIPT'
        return res