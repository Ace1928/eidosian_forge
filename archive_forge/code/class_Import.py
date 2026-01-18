important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class Import(PythonBaseNode):
    __slots__ = ()

    def get_path_for_name(self, name):
        """
        The path is the list of names that leads to the searched name.

        :return list of Name:
        """
        try:
            name = self._aliases()[name]
        except KeyError:
            pass
        for path in self.get_paths():
            if name in path:
                return path[:path.index(name) + 1]
        raise ValueError('Name should be defined in the import itself')

    def is_nested(self):
        return False

    def is_star_import(self):
        return self.children[-1] == '*'