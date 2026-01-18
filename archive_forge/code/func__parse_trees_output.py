import os
import tempfile
import warnings
from subprocess import PIPE
from nltk.internals import (
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tree import Tree
def _parse_trees_output(self, output_):
    res = []
    cur_lines = []
    cur_trees = []
    blank = False
    for line in output_.splitlines(False):
        if line == '':
            if blank:
                res.append(iter(cur_trees))
                cur_trees = []
                blank = False
            elif self._DOUBLE_SPACED_OUTPUT:
                cur_trees.append(self._make_tree('\n'.join(cur_lines)))
                cur_lines = []
                blank = True
            else:
                res.append(iter([self._make_tree('\n'.join(cur_lines))]))
                cur_lines = []
        else:
            cur_lines.append(line)
            blank = False
    return iter(res)