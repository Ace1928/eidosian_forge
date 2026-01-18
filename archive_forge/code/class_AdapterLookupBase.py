import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
class AdapterLookupBase:

    def __init__(self, registry):
        self._registry = registry
        self._required = {}
        self.init_extendors()
        super().__init__()

    def changed(self, ignored=None):
        super().changed(None)
        for r in self._required.keys():
            r = r()
            if r is not None:
                r.unsubscribe(self)
        self._required.clear()

    def init_extendors(self):
        self._extendors = {}
        for p in self._registry._provided:
            self.add_extendor(p)

    def add_extendor(self, provided):
        _extendors = self._extendors
        for i in provided.__iro__:
            extendors = _extendors.get(i, ())
            _extendors[i] = [e for e in extendors if provided.isOrExtends(e)] + [provided] + [e for e in extendors if not provided.isOrExtends(e)]

    def remove_extendor(self, provided):
        _extendors = self._extendors
        for i in provided.__iro__:
            _extendors[i] = [e for e in _extendors.get(i, ()) if e != provided]

    def _subscribe(self, *required):
        _refs = self._required
        for r in required:
            ref = r.weakref()
            if ref not in _refs:
                r.subscribe(self)
                _refs[ref] = 1

    def _uncached_lookup(self, required, provided, name=''):
        required = tuple(required)
        result = None
        order = len(required)
        for registry in self._registry.ro:
            byorder = registry._adapters
            if order >= len(byorder):
                continue
            extendors = registry._v_lookup._extendors.get(provided)
            if not extendors:
                continue
            components = byorder[order]
            result = _lookup(components, required, extendors, name, 0, order)
            if result is not None:
                break
        self._subscribe(*required)
        return result

    def queryMultiAdapter(self, objects, provided, name='', default=None):
        factory = self.lookup([providedBy(o) for o in objects], provided, name)
        if factory is None:
            return default
        result = factory(*[o.__self__ if isinstance(o, super) else o for o in objects])
        if result is None:
            return default
        return result

    def _uncached_lookupAll(self, required, provided):
        required = tuple(required)
        order = len(required)
        result = {}
        for registry in reversed(self._registry.ro):
            byorder = registry._adapters
            if order >= len(byorder):
                continue
            extendors = registry._v_lookup._extendors.get(provided)
            if not extendors:
                continue
            components = byorder[order]
            _lookupAll(components, required, extendors, result, 0, order)
        self._subscribe(*required)
        return tuple(result.items())

    def names(self, required, provided):
        return [c[0] for c in self.lookupAll(required, provided)]

    def _uncached_subscriptions(self, required, provided):
        required = tuple(required)
        order = len(required)
        result = []
        for registry in reversed(self._registry.ro):
            byorder = registry._subscribers
            if order >= len(byorder):
                continue
            if provided is None:
                extendors = (provided,)
            else:
                extendors = registry._v_lookup._extendors.get(provided)
                if extendors is None:
                    continue
            _subscriptions(byorder[order], required, extendors, '', result, 0, order)
        self._subscribe(*required)
        return result

    def subscribers(self, objects, provided):
        subscriptions = self.subscriptions([providedBy(o) for o in objects], provided)
        if provided is None:
            result = ()
            for subscription in subscriptions:
                subscription(*objects)
        else:
            result = []
            for subscription in subscriptions:
                subscriber = subscription(*objects)
                if subscriber is not None:
                    result.append(subscriber)
        return result