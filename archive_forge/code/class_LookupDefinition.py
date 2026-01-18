from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class LookupDefinition(Statement):

    def __init__(self, name, process_base, process_marks, mark_glyph_set, direction, reversal, comments, context, sub, pos, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.process_base = process_base
        self.process_marks = process_marks
        self.mark_glyph_set = mark_glyph_set
        self.direction = direction
        self.reversal = reversal
        self.comments = comments
        self.context = context
        self.sub = sub
        self.pos = pos

    def __str__(self):
        res = f'DEF_LOOKUP "{self.name}"'
        res += f' {self.process_base and 'PROCESS_BASE' or 'SKIP_BASE'}'
        if self.process_marks:
            res += ' PROCESS_MARKS '
            if self.mark_glyph_set:
                res += f'MARK_GLYPH_SET "{self.mark_glyph_set}"'
            elif isinstance(self.process_marks, str):
                res += f'"{self.process_marks}"'
            else:
                res += 'ALL'
        else:
            res += ' SKIP_MARKS'
        if self.direction is not None:
            res += f' DIRECTION {self.direction}'
        if self.reversal:
            res += ' REVERSAL'
        if self.comments is not None:
            comments = self.comments.replace('\n', '\\n')
            res += f'\nCOMMENTS "{comments}"'
        if self.context:
            res += '\n' + '\n'.join((str(c) for c in self.context))
        else:
            res += '\nIN_CONTEXT\nEND_CONTEXT'
        if self.sub:
            res += f'\n{self.sub}'
        if self.pos:
            res += f'\n{self.pos}'
        return res