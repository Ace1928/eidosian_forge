import weakref
from .. import util
from ..orm import attributes
from ..orm import base as orm_base
from ..orm import collections
from ..orm import exc as orm_exc
from ..orm import instrumentation as orm_instrumentation
from ..orm import util as orm_util
from ..orm.instrumentation import _default_dict_getter
from ..orm.instrumentation import _default_manager_getter
from ..orm.instrumentation import _default_opt_manager_getter
from ..orm.instrumentation import _default_state_getter
from ..orm.instrumentation import ClassManager
from ..orm.instrumentation import InstrumentationFactory
class ExtendedInstrumentationRegistry(InstrumentationFactory):
    """Extends :class:`.InstrumentationFactory` with additional
    bookkeeping, to accommodate multiple types of
    class managers.

    """
    _manager_finders = weakref.WeakKeyDictionary()
    _state_finders = weakref.WeakKeyDictionary()
    _dict_finders = weakref.WeakKeyDictionary()
    _extended = False

    def _locate_extended_factory(self, class_):
        for finder in instrumentation_finders:
            factory = finder(class_)
            if factory is not None:
                manager = self._extended_class_manager(class_, factory)
                return (manager, factory)
        else:
            return (None, None)

    def _check_conflicts(self, class_, factory):
        existing_factories = self._collect_management_factories_for(class_).difference([factory])
        if existing_factories:
            raise TypeError('multiple instrumentation implementations specified in %s inheritance hierarchy: %r' % (class_.__name__, list(existing_factories)))

    def _extended_class_manager(self, class_, factory):
        manager = factory(class_)
        if not isinstance(manager, ClassManager):
            manager = _ClassInstrumentationAdapter(class_, manager)
        if factory != ClassManager and (not self._extended):
            self._extended = True
            _install_instrumented_lookups()
        self._manager_finders[class_] = manager.manager_getter()
        self._state_finders[class_] = manager.state_getter()
        self._dict_finders[class_] = manager.dict_getter()
        return manager

    def _collect_management_factories_for(self, cls):
        """Return a collection of factories in play or specified for a
        hierarchy.

        Traverses the entire inheritance graph of a cls and returns a
        collection of instrumentation factories for those classes. Factories
        are extracted from active ClassManagers, if available, otherwise
        instrumentation_finders is consulted.

        """
        hierarchy = util.class_hierarchy(cls)
        factories = set()
        for member in hierarchy:
            manager = self.opt_manager_of_class(member)
            if manager is not None:
                factories.add(manager.factory)
            else:
                for finder in instrumentation_finders:
                    factory = finder(member)
                    if factory is not None:
                        break
                else:
                    factory = None
                factories.add(factory)
        factories.discard(None)
        return factories

    def unregister(self, class_):
        super().unregister(class_)
        if class_ in self._manager_finders:
            del self._manager_finders[class_]
            del self._state_finders[class_]
            del self._dict_finders[class_]

    def opt_manager_of_class(self, cls):
        try:
            finder = self._manager_finders.get(cls, _default_opt_manager_getter)
        except TypeError:
            return None
        else:
            return finder(cls)

    def manager_of_class(self, cls):
        try:
            finder = self._manager_finders.get(cls, _default_manager_getter)
        except TypeError:
            raise orm_exc.UnmappedClassError(cls, f"Can't locate an instrumentation manager for class {cls}")
        else:
            manager = finder(cls)
            if manager is None:
                raise orm_exc.UnmappedClassError(cls, f"Can't locate an instrumentation manager for class {cls}")
            return manager

    def state_of(self, instance):
        if instance is None:
            raise AttributeError('None has no persistent state.')
        return self._state_finders.get(instance.__class__, _default_state_getter)(instance)

    def dict_of(self, instance):
        if instance is None:
            raise AttributeError('None has no persistent state.')
        return self._dict_finders.get(instance.__class__, _default_dict_getter)(instance)