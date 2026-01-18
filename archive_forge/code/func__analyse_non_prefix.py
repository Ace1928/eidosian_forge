import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
def _analyse_non_prefix(self, leaf):
    typ = leaf.type
    if typ == 'name' and leaf.value in ('l', 'O', 'I'):
        if leaf.is_definition():
            message = "Do not define %s named 'l', 'O', or 'I' one line"
            if leaf.parent.type == 'class' and leaf.parent.name == leaf:
                self.add_issue(leaf, 742, message % 'classes')
            elif leaf.parent.type == 'function' and leaf.parent.name == leaf:
                self.add_issue(leaf, 743, message % 'function')
            else:
                self.add_issuadd_issue(741, message % 'variables', leaf)
    elif leaf.value == ':':
        if isinstance(leaf.parent, (Flow, Scope)) and leaf.parent.type != 'lambdef':
            next_leaf = leaf.get_next_leaf()
            if next_leaf.type != 'newline':
                if leaf.parent.type == 'funcdef':
                    self.add_issue(next_leaf, 704, 'Multiple statements on one line (def)')
                else:
                    self.add_issue(next_leaf, 701, 'Multiple statements on one line (colon)')
    elif leaf.value == ';':
        if leaf.get_next_leaf().type in ('newline', 'endmarker'):
            self.add_issue(leaf, 703, 'Statement ends with a semicolon')
        else:
            self.add_issue(leaf, 702, 'Multiple statements on one line (semicolon)')
    elif leaf.value in ('==', '!='):
        comparison = leaf.parent
        index = comparison.children.index(leaf)
        left = comparison.children[index - 1]
        right = comparison.children[index + 1]
        for node in (left, right):
            if node.type == 'keyword' or node.type == 'name':
                if node.value == 'None':
                    message = "comparison to None should be 'if cond is None:'"
                    self.add_issue(leaf, 711, message)
                    break
                elif node.value in ('True', 'False'):
                    message = "comparison to False/True should be 'if cond is True:' or 'if cond:'"
                    self.add_issue(leaf, 712, message)
                    break
    elif leaf.value in ('in', 'is'):
        comparison = leaf.parent
        if comparison.type == 'comparison' and comparison.parent.type == 'not_test':
            if leaf.value == 'in':
                self.add_issue(leaf, 713, "test for membership should be 'not in'")
            else:
                self.add_issue(leaf, 714, "test for object identity should be 'is not'")
    elif typ == 'string':
        for i, line in enumerate(leaf.value.splitlines()[1:]):
            indentation = re.match('[ \\t]*', line).group(0)
            start_pos = (leaf.line + i, len(indentation))
            start_pos
    elif typ == 'endmarker':
        if self._newline_count >= 2:
            self.add_issue(leaf, 391, 'Blank line at end of file')