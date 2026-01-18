import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
class _Jellier:
    """
    (Internal) This class manages state for a call to jelly()
    """

    def __init__(self, taster, persistentStore, invoker):
        """
        Initialize.
        """
        self.taster = taster
        self.preserved = {}
        self.cooked = {}
        self.cooker = {}
        self._ref_id = 1
        self.persistentStore = persistentStore
        self.invoker = invoker

    def _cook(self, object):
        """
        (internal) Backreference an object.

        Notes on this method for the hapless future maintainer: If I've already
        gone through the prepare/preserve cycle on the specified object (it is
        being referenced after the serializer is "done with" it, e.g. this
        reference is NOT circular), the copy-in-place of aList is relevant,
        since the list being modified is the actual, pre-existing jelly
        expression that was returned for that object. If not, it's technically
        superfluous, since the value in self.preserved didn't need to be set,
        but the invariant that self.preserved[id(object)] is a list is
        convenient because that means we don't have to test and create it or
        not create it here, creating fewer code-paths.  that's why
        self.preserved is always set to a list.

        Sorry that this code is so hard to follow, but Python objects are
        tricky to persist correctly. -glyph
        """
        aList = self.preserved[id(object)]
        newList = copy.copy(aList)
        refid = self._ref_id
        self._ref_id = self._ref_id + 1
        aList[:] = [reference_atom, refid, newList]
        self.cooked[id(object)] = [dereference_atom, refid]
        return aList

    def prepare(self, object):
        """
        (internal) Create a list for persisting an object to.  This will allow
        backreferences to be made internal to the object. (circular
        references).

        The reason this needs to happen is that we don't generate an ID for
        every object, so we won't necessarily know which ID the object will
        have in the future.  When it is 'cooked' ( see _cook ), it will be
        assigned an ID, and the temporary placeholder list created here will be
        modified in-place to create an expression that gives this object an ID:
        [reference id# [object-jelly]].
        """
        self.preserved[id(object)] = []
        self.cooker[id(object)] = object
        return []

    def preserve(self, object, sexp):
        """
        (internal) Mark an object's persistent list for later referral.
        """
        if id(object) in self.cooked:
            self.preserved[id(object)][2] = sexp
            sexp = self.preserved[id(object)]
        else:
            self.preserved[id(object)] = sexp
        return sexp

    def _checkMutable(self, obj):
        objId = id(obj)
        if objId in self.cooked:
            return self.cooked[objId]
        if objId in self.preserved:
            self._cook(obj)
            return self.cooked[objId]

    def jelly(self, obj):
        if isinstance(obj, Jellyable):
            preRef = self._checkMutable(obj)
            if preRef:
                return preRef
            return obj.jellyFor(self)
        objType = type(obj)
        if self.taster.isTypeAllowed(qual(objType).encode('utf-8')):
            if objType in (bytes, int, float):
                return obj
            elif isinstance(obj, types.MethodType):
                aSelf = obj.__self__
                aFunc = obj.__func__
                aClass = aSelf.__class__
                return [b'method', aFunc.__name__, self.jelly(aSelf), self.jelly(aClass)]
            elif objType is str:
                return [b'unicode', obj.encode('UTF-8')]
            elif isinstance(obj, type(None)):
                return [b'None']
            elif isinstance(obj, types.FunctionType):
                return [b'function', obj.__module__ + '.' + obj.__qualname__]
            elif isinstance(obj, types.ModuleType):
                return [b'module', obj.__name__]
            elif objType is bool:
                return [b'boolean', obj and b'true' or b'false']
            elif objType is datetime.datetime:
                if obj.tzinfo:
                    raise NotImplementedError("Currently can't jelly datetime objects with tzinfo")
                return [b'datetime', ' '.join([str(x) for x in (obj.year, obj.month, obj.day, obj.hour, obj.minute, obj.second, obj.microsecond)]).encode('utf-8')]
            elif objType is datetime.time:
                if obj.tzinfo:
                    raise NotImplementedError("Currently can't jelly datetime objects with tzinfo")
                return [b'time', ' '.join([str(x) for x in (obj.hour, obj.minute, obj.second, obj.microsecond)]).encode('utf-8')]
            elif objType is datetime.date:
                return [b'date', ' '.join([str(x) for x in (obj.year, obj.month, obj.day)]).encode('utf-8')]
            elif objType is datetime.timedelta:
                return [b'timedelta', ' '.join([str(x) for x in (obj.days, obj.seconds, obj.microseconds)]).encode('utf-8')]
            elif issubclass(objType, type):
                return [b'class', qual(obj).encode('utf-8')]
            elif objType is decimal.Decimal:
                return self.jelly_decimal(obj)
            else:
                preRef = self._checkMutable(obj)
                if preRef:
                    return preRef
                sxp = self.prepare(obj)
                if objType is list:
                    sxp.extend(self._jellyIterable(list_atom, obj))
                elif objType is tuple:
                    sxp.extend(self._jellyIterable(tuple_atom, obj))
                elif objType in DictTypes:
                    sxp.append(dictionary_atom)
                    for key, val in obj.items():
                        sxp.append([self.jelly(key), self.jelly(val)])
                elif objType is set:
                    sxp.extend(self._jellyIterable(set_atom, obj))
                elif objType is frozenset:
                    sxp.extend(self._jellyIterable(frozenset_atom, obj))
                else:
                    className = qual(obj.__class__).encode('utf-8')
                    persistent = None
                    if self.persistentStore:
                        persistent = self.persistentStore(obj, self)
                    if persistent is not None:
                        sxp.append(persistent_atom)
                        sxp.append(persistent)
                    elif self.taster.isClassAllowed(obj.__class__):
                        sxp.append(className)
                        if hasattr(obj, '__getstate__'):
                            state = obj.__getstate__()
                        else:
                            state = obj.__dict__
                        sxp.append(self.jelly(state))
                    else:
                        self.unpersistable('instance of class %s deemed insecure' % qual(obj.__class__), sxp)
                return self.preserve(obj, sxp)
        else:
            raise InsecureJelly(f'Type not allowed for object: {objType} {obj}')

    def _jellyIterable(self, atom, obj):
        """
        Jelly an iterable object.

        @param atom: the identifier atom of the object.
        @type atom: C{str}

        @param obj: any iterable object.
        @type obj: C{iterable}

        @return: a generator of jellied data.
        @rtype: C{generator}
        """
        yield atom
        for item in obj:
            yield self.jelly(item)

    def jelly_decimal(self, d):
        """
        Jelly a decimal object.

        @param d: a decimal object to serialize.
        @type d: C{decimal.Decimal}

        @return: jelly for the decimal object.
        @rtype: C{list}
        """
        sign, guts, exponent = d.as_tuple()
        value = reduce(lambda left, right: left * 10 + right, guts)
        if sign:
            value = -value
        return [b'decimal', value, exponent]

    def unpersistable(self, reason, sxp=None):
        """
        (internal) Returns an sexp: (unpersistable "reason").  Utility method
        for making note that a particular object could not be serialized.
        """
        if sxp is None:
            sxp = []
        sxp.append(unpersistable_atom)
        if isinstance(reason, str):
            reason = reason.encode('utf-8')
        sxp.append(reason)
        return sxp