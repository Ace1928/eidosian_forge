import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_SwitchStatement(self, node: ast.SwitchStatement) -> None:
    self._start_line()
    self.stream.write('switch (')
    self.visit(node.target)
    self.stream.write(') {')
    self._end_line()
    self._current_indent += 1
    for labels, case in node.cases:
        if not labels:
            continue
        self._start_line()
        self.stream.write('case ')
        self._visit_sequence(labels, separator=', ')
        self.stream.write(' ')
        self.visit(case)
        self._end_line()
    if node.default is not None:
        self._start_line()
        self.stream.write('default ')
        self.visit(node.default)
        self._end_line()
    self._current_indent -= 1
    self._start_line()
    self.stream.write('}')
    self._end_line()