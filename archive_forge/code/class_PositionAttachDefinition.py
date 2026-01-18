from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class PositionAttachDefinition(Statement):

    def __init__(self, coverage, coverage_to, location=None):
        Statement.__init__(self, location)
        self.coverage = coverage
        self.coverage_to = coverage_to

    def __str__(self):
        coverage = ''.join((str(c) for c in self.coverage))
        res = f'AS_POSITION\nATTACH{coverage}\nTO'
        for coverage, anchor in self.coverage_to:
            coverage = ''.join((str(c) for c in coverage))
            res += f'{coverage} AT ANCHOR "{anchor}"'
        res += '\nEND_ATTACH\nEND_POSITION'
        return res