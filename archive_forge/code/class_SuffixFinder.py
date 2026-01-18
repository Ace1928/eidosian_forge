import enum
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import In
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import Initializer
class SuffixFinder(object):

    def __init__(self, name, default=None):
        """This provides an efficient utility for finding suffix values on a
        (hierarchical) Pyomo model.

        Parameters
        ----------
        name: str

            Name of Suffix to search for.

        default:

            Default value to return from `.find()` if no matching Suffix
            is found.

        """
        self.name = name
        self.default = default
        self.all_suffixes = []
        self._suffixes_by_block = {None: []}

    def find(self, component_data):
        """Find suffix value for a given component data object in model tree

        Suffixes are searched by traversing the model hierarchy in three passes:

        1. Search for a Suffix matching the specific component_data,
           starting at the `root` and descending down the tree to
           the component_data.  Return the first match found.
        2. Search for a Suffix matching the component_data's container,
           starting at the `root` and descending down the tree to
           the component_data.  Return the first match found.
        3. Search for a Suffix with key `None`, starting from the
           component_data and working up the tree to the `root`.
           Return the first match found.
        4. Return the default value

        Parameters
        ----------
        component_data: ComponentDataBase

            Component or component data object to find suffix value for.

        Returns
        -------
        The value for Suffix associated with component data if found, else None.

        """
        suffixes = self._get_suffix_list(component_data.parent_block())
        for s in suffixes:
            if component_data in s:
                return s[component_data]
        parent_comp = component_data.parent_component()
        if parent_comp is not component_data:
            for s in suffixes:
                if parent_comp in s:
                    return s[parent_comp]
        for s in reversed(suffixes):
            if None in s:
                return s[None]
        return self.default

    def _get_suffix_list(self, parent):
        if parent in self._suffixes_by_block:
            return self._suffixes_by_block[parent]
        suffixes = list(self._get_suffix_list(parent.parent_block()))
        self._suffixes_by_block[parent] = suffixes
        s = parent.component(self.name)
        if s is not None and s.ctype is Suffix and s.active:
            suffixes.append(s)
            self.all_suffixes.append(s)
        return suffixes