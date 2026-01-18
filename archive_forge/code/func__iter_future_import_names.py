important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
def _iter_future_import_names(self):
    """
        :return: A list of future import names.
        :rtype: list of str
        """
    for imp in self.iter_imports():
        if imp.type == 'import_from' and imp.level == 0:
            for path in imp.get_paths():
                names = [name.value for name in path]
                if len(names) == 2 and names[0] == '__future__':
                    yield names[1]