from contextlib import contextmanager
from typing import Dict, List
class RefactoringNormalizer(Normalizer):

    def __init__(self, node_to_str_map):
        self._node_to_str_map = node_to_str_map

    def visit(self, node):
        try:
            return self._node_to_str_map[node]
        except KeyError:
            return super().visit(node)

    def visit_leaf(self, leaf):
        try:
            return self._node_to_str_map[leaf]
        except KeyError:
            return super().visit_leaf(leaf)