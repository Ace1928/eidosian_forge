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
@ModelComponentFactory.register('Declare a container for extraneous model data')
class Suffix(ComponentMap, ActiveComponent):
    """A model suffix, representing extraneous model data"""
    '\n    Constructor Arguments:\n        direction   The direction of information flow for this suffix.\n                        By default, this is LOCAL, indicating that no\n                        suffix data is exported or imported.\n        datatype    A variable type associated with all values of this\n                        suffix.\n    '
    LOCAL = SuffixDirection.LOCAL
    EXPORT = SuffixDirection.EXPORT
    IMPORT = SuffixDirection.IMPORT
    IMPORT_EXPORT = SuffixDirection.IMPORT_EXPORT
    FLOAT = SuffixDataType.FLOAT
    INT = SuffixDataType.INT

    def __new__(cls, *args, **kwargs):
        if cls is not Suffix:
            return super().__new__(cls)
        return super().__new__(AbstractSuffix)

    def __setstate__(self, state):
        super().__setstate__(state)
        if self._constructed and self.__class__ is AbstractSuffix:
            self.__class__ = Suffix

    @overload
    def __init__(self, *, direction=LOCAL, datatype=FLOAT, initialize=None, rule=None, name=None, doc=None):
        ...

    def __init__(self, **kwargs):
        self._direction = None
        self._datatype = None
        self._rule = None
        self.direction = kwargs.pop('direction', Suffix.LOCAL)
        self.datatype = kwargs.pop('datatype', Suffix.FLOAT)
        self._rule = Initializer(self._pop_from_kwargs('Suffix', kwargs, ('rule', 'initialize'), None), treat_sequences_as_mappings=False, allow_generators=True)
        kwargs.setdefault('ctype', Suffix)
        ActiveComponent.__init__(self, **kwargs)
        ComponentMap.__init__(self)
        if self._rule is None:
            self.construct()

    def construct(self, data=None):
        """
        Constructs this component, applying rule if it exists.
        """
        if is_debug_set(logger):
            logger.debug(f"Constructing %s '%s'", self.__class__.__name__, self.name)
        if self._constructed is True:
            return
        timer = ConstructionTimer(self)
        self._constructed = True
        if self._rule is not None:
            rule = self._rule
            if rule.contains_indices():
                block = self.parent_block()
                for index in rule.indices():
                    self.set_value(index, rule(block, index), expand=True)
            else:
                self.update_values(rule(self.parent_block(), None), expand=True)
        timer.report()

    @property
    def datatype(self):
        """Return the suffix datatype."""
        return self._datatype

    @datatype.setter
    def datatype(self, datatype):
        """Set the suffix datatype."""
        if datatype is not None:
            datatype = _SuffixDataTypeDomain(datatype)
        self._datatype = datatype

    @property
    def direction(self):
        """Return the suffix direction."""
        return self._direction

    @direction.setter
    def direction(self, direction):
        """Set the suffix direction."""
        self._direction = _SuffixDirectionDomain(direction)

    def export_enabled(self):
        """
        Returns True when this suffix is enabled for export to
        solvers.
        """
        return bool(self._direction & Suffix.EXPORT)

    def import_enabled(self):
        """
        Returns True when this suffix is enabled for import from
        solutions.
        """
        return bool(self._direction & Suffix.IMPORT)

    def update_values(self, data, expand=True):
        """
        Updates the suffix data given a list of component,value
        tuples. Provides an improvement in efficiency over calling
        set_value on every component.
        """
        if expand:
            try:
                items = data.items()
            except AttributeError:
                items = data
            for component, value in items:
                self.set_value(component, value, expand=expand)
        else:
            self.update(data)

    def set_value(self, component, value, expand=True):
        """
        Sets the value of this suffix on the specified component.

        When expand is True (default), array components are handled by
        storing a reference and value for each index, with no
        reference being stored for the array component itself. When
        expand is False (this is the case for __setitem__), this
        behavior is disabled and a reference to the array component
        itself is kept.
        """
        if expand and component.is_indexed():
            for component_ in component.values():
                self[component_] = value
        else:
            self[component] = value

    def set_all_values(self, value):
        """
        Sets the value of this suffix on all components.
        """
        for ndx in self:
            self[ndx] = value

    def clear_value(self, component, expand=True):
        """
        Clears suffix information for a component.
        """
        if expand and component.is_indexed():
            for component_ in component.values():
                self.pop(component_, None)
        else:
            self.pop(component, None)

    def clear_all_values(self):
        """
        Clears all suffix data.
        """
        self.clear()

    @deprecated('Suffix.set_datatype is replaced with the Suffix.datatype property', version='6.7.1')
    def set_datatype(self, datatype):
        """
        Set the suffix datatype.
        """
        self.datatype = datatype

    @deprecated('Suffix.get_datatype is replaced with the Suffix.datatype property', version='6.7.1')
    def get_datatype(self):
        """
        Return the suffix datatype.
        """
        return self.datatype

    @deprecated('Suffix.set_direction is replaced with the Suffix.direction property', version='6.7.1')
    def set_direction(self, direction):
        """
        Set the suffix direction.
        """
        self.direction = direction

    @deprecated('Suffix.get_direction is replaced with the Suffix.direction property', version='6.7.1')
    def get_direction(self):
        """
        Return the suffix direction.
        """
        return self.direction

    def _pprint(self):
        return ([('Direction', str(self._direction.name)), ('Datatype', getattr(self._datatype, 'name', 'None'))], ((str(k), v) for k, v in self._dict.values()), ('Value',), lambda k, v: [v])

    def pprint(self, *args, **kwds):
        return ActiveComponent.pprint(self, *args, **kwds)

    def __str__(self):
        return ActiveComponent.__str__(self)