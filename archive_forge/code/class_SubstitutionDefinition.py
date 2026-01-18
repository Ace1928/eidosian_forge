from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class SubstitutionDefinition(Statement):

    def __init__(self, mapping, location=None):
        Statement.__init__(self, location)
        self.mapping = mapping

    def __str__(self):
        res = 'AS_SUBSTITUTION\n'
        for src, dst in self.mapping.items():
            src = ''.join((str(s) for s in src))
            dst = ''.join((str(d) for d in dst))
            res += f'SUB{src}\nWITH{dst}\nEND_SUB\n'
        res += 'END_SUBSTITUTION'
        return res