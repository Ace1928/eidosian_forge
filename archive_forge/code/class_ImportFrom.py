important if you want to refactor a parser tree.
import re
from typing import Tuple
from parso.tree import Node, BaseNode, Leaf, ErrorNode, ErrorLeaf, search_ancestor  # noqa
from parso.python.prefix import split_prefix
from parso.utils import split_lines
class ImportFrom(Import):
    type = 'import_from'
    __slots__ = ()

    def get_defined_names(self, include_setitem=False):
        """
        Returns the a list of `Name` that the import defines. The
        defined names are set after `import` or in case an alias - `as` - is
        present that name is returned.
        """
        return [alias or name for name, alias in self._as_name_tuples()]

    def _aliases(self):
        """Mapping from alias to its corresponding name."""
        return dict(((alias, name) for name, alias in self._as_name_tuples() if alias is not None))

    def get_from_names(self):
        for n in self.children[1:]:
            if n not in ('.', '...'):
                break
        if n.type == 'dotted_name':
            return n.children[::2]
        elif n == 'import':
            return []
        else:
            return [n]

    @property
    def level(self):
        """The level parameter of ``__import__``."""
        level = 0
        for n in self.children[1:]:
            if n in ('.', '...'):
                level += len(n.value)
            else:
                break
        return level

    def _as_name_tuples(self):
        last = self.children[-1]
        if last == ')':
            last = self.children[-2]
        elif last == '*':
            return
        if last.type == 'import_as_names':
            as_names = last.children[::2]
        else:
            as_names = [last]
        for as_name in as_names:
            if as_name.type == 'name':
                yield (as_name, None)
            else:
                yield as_name.children[::2]

    def get_paths(self):
        """
        The import paths defined in an import statement. Typically an array
        like this: ``[<Name: datetime>, <Name: date>]``.

        :return list of list of Name:
        """
        dotted = self.get_from_names()
        if self.children[-1] == '*':
            return [dotted]
        return [dotted + [name] for name, alias in self._as_name_tuples()]