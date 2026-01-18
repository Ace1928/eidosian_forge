from __future__ import absolute_import
import warnings
import textwrap
from ruamel.yaml.compat import utf8
class MarkedYAMLFutureWarning(YAMLFutureWarning):

    def __init__(self, context=None, context_mark=None, problem=None, problem_mark=None, note=None, warn=None):
        self.context = context
        self.context_mark = context_mark
        self.problem = problem
        self.problem_mark = problem_mark
        self.note = note
        self.warn = warn

    def __str__(self):
        lines = []
        if self.context is not None:
            lines.append(self.context)
        if self.context_mark is not None and (self.problem is None or self.problem_mark is None or self.context_mark.name != self.problem_mark.name or (self.context_mark.line != self.problem_mark.line) or (self.context_mark.column != self.problem_mark.column)):
            lines.append(str(self.context_mark))
        if self.problem is not None:
            lines.append(self.problem)
        if self.problem_mark is not None:
            lines.append(str(self.problem_mark))
        if self.note is not None and self.note:
            note = textwrap.dedent(self.note)
            lines.append(note)
        if self.warn is not None and self.warn:
            warn = textwrap.dedent(self.warn)
            lines.append(warn)
        return '\n'.join(lines)