import re
from contextlib import contextmanager
from typing import Tuple
from parso.python.errors import ErrorFinder, ErrorFinderConfig
from parso.normalizer import Rule
from parso.python.tree import Flow, Scope
class IndentationNode(object):
    type = IndentationTypes.SUITE

    def __init__(self, config, indentation, parent=None):
        self.bracket_indentation = self.indentation = indentation
        self.parent = parent

    def __repr__(self):
        return '<%s>' % self.__class__.__name__

    def get_latest_suite_node(self):
        n = self
        while n is not None:
            if n.type == IndentationTypes.SUITE:
                return n
            n = n.parent