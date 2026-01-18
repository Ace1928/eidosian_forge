import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_SwitchStatementPreview(self, node: ast.SwitchStatementPreview) -> None:
    self._start_line()
    self.stream.write('switch (')
    self.visit(node.target)
    self.stream.write(') {')
    self._end_line()
    self._current_indent += 1
    for labels, case in node.cases:
        if not labels:
            continue
        for label in labels[:-1]:
            self._start_line()
            self.stream.write('case ')
            self.visit(label)
            self.stream.write(':')
            self._end_line()
        self._start_line()
        if isinstance(labels[-1], ast.DefaultCase):
            self.visit(labels[-1])
        else:
            self.stream.write('case ')
            self.visit(labels[-1])
        self.stream.write(': ')
        self.visit(case)
        self._end_line()
        self._write_statement('break')
    self._current_indent -= 1
    self._start_line()
    self.stream.write('}')
    self._end_line()