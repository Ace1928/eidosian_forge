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
class _Unjellier:

    def __init__(self, taster, persistentLoad, invoker):
        self.taster = taster
        self.persistentLoad = persistentLoad
        self.references = {}
        self.postCallbacks = []
        self.invoker = invoker

    def unjellyFull(self, obj):
        o = self.unjelly(obj)
        for m in self.postCallbacks:
            m()
        return o

    def _maybePostUnjelly(self, unjellied):
        """
        If the given object has support for the C{postUnjelly} hook, set it up
        to be called at the end of deserialization.

        @param unjellied: an object that has already been unjellied.

        @return: C{unjellied}
        """
        if hasattr(unjellied, 'postUnjelly'):
            self.postCallbacks.append(unjellied.postUnjelly)
        return unjellied

    def unjelly(self, obj):
        if type(obj) is not list:
            return obj
        jelTypeBytes = obj[0]
        if not self.taster.isTypeAllowed(jelTypeBytes):
            raise InsecureJelly(jelTypeBytes)
        regClass = unjellyableRegistry.get(jelTypeBytes)
        if regClass is not None:
            method = getattr(_createBlank(regClass), 'unjellyFor', regClass)
            return self._maybePostUnjelly(method(self, obj))
        regFactory = unjellyableFactoryRegistry.get(jelTypeBytes)
        if regFactory is not None:
            return self._maybePostUnjelly(regFactory(self.unjelly(obj[1])))
        jelTypeText = nativeString(jelTypeBytes)
        thunk = getattr(self, '_unjelly_%s' % jelTypeText, None)
        if thunk is not None:
            return thunk(obj[1:])
        else:
            nameSplit = jelTypeText.split('.')
            modName = '.'.join(nameSplit[:-1])
            if not self.taster.isModuleAllowed(modName):
                raise InsecureJelly(f'Module {modName} not allowed (in type {jelTypeText}).')
            clz = namedObject(jelTypeText)
            if not self.taster.isClassAllowed(clz):
                raise InsecureJelly('Class %s not allowed.' % jelTypeText)
            return self._genericUnjelly(clz, obj[1])

    def _genericUnjelly(self, cls, state):
        """
        Unjelly a type for which no specific unjellier is registered, but which
        is nonetheless allowed.

        @param cls: the class of the instance we are unjellying.
        @type cls: L{type}

        @param state: The jellied representation of the object's state; its
            C{__dict__} unless it has a C{__setstate__} that takes something
            else.
        @type state: L{list}

        @return: the new, unjellied instance.
        """
        return self._maybePostUnjelly(_newInstance(cls, self.unjelly(state)))

    def _unjelly_None(self, exp):
        return None

    def _unjelly_unicode(self, exp):
        return str(exp[0], 'UTF-8')

    def _unjelly_decimal(self, exp):
        """
        Unjelly decimal objects.
        """
        value = exp[0]
        exponent = exp[1]
        if value < 0:
            sign = 1
        else:
            sign = 0
        guts = decimal.Decimal(value).as_tuple()[1]
        return decimal.Decimal((sign, guts, exponent))

    def _unjelly_boolean(self, exp):
        assert exp[0] in (b'true', b'false')
        return exp[0] == b'true'

    def _unjelly_datetime(self, exp):
        return datetime.datetime(*map(int, exp[0].split()))

    def _unjelly_date(self, exp):
        return datetime.date(*map(int, exp[0].split()))

    def _unjelly_time(self, exp):
        return datetime.time(*map(int, exp[0].split()))

    def _unjelly_timedelta(self, exp):
        days, seconds, microseconds = map(int, exp[0].split())
        return datetime.timedelta(days=days, seconds=seconds, microseconds=microseconds)

    def unjellyInto(self, obj, loc, jel):
        o = self.unjelly(jel)
        if isinstance(o, NotKnown):
            o.addDependant(obj, loc)
        obj[loc] = o
        return o

    def _unjelly_dereference(self, lst):
        refid = lst[0]
        x = self.references.get(refid)
        if x is not None:
            return x
        der = _Dereference(refid)
        self.references[refid] = der
        return der

    def _unjelly_reference(self, lst):
        refid = lst[0]
        exp = lst[1]
        o = self.unjelly(exp)
        ref = self.references.get(refid)
        if ref is None:
            self.references[refid] = o
        elif isinstance(ref, NotKnown):
            ref.resolveDependants(o)
            self.references[refid] = o
        else:
            assert 0, 'Multiple references with same ID!'
        return o

    def _unjelly_tuple(self, lst):
        l = list(range(len(lst)))
        finished = 1
        for elem in l:
            if isinstance(self.unjellyInto(l, elem, lst[elem]), NotKnown):
                finished = 0
        if finished:
            return tuple(l)
        else:
            return _Tuple(l)

    def _unjelly_list(self, lst):
        l = list(range(len(lst)))
        for elem in l:
            self.unjellyInto(l, elem, lst[elem])
        return l

    def _unjellySetOrFrozenset(self, lst, containerType):
        """
        Helper method to unjelly set or frozenset.

        @param lst: the content of the set.
        @type lst: C{list}

        @param containerType: the type of C{set} to use.
        """
        l = list(range(len(lst)))
        finished = True
        for elem in l:
            data = self.unjellyInto(l, elem, lst[elem])
            if isinstance(data, NotKnown):
                finished = False
        if not finished:
            return _Container(l, containerType)
        else:
            return containerType(l)

    def _unjelly_set(self, lst):
        """
        Unjelly set using the C{set} builtin.
        """
        return self._unjellySetOrFrozenset(lst, set)

    def _unjelly_frozenset(self, lst):
        """
        Unjelly frozenset using the C{frozenset} builtin.
        """
        return self._unjellySetOrFrozenset(lst, frozenset)

    def _unjelly_dictionary(self, lst):
        d = {}
        for k, v in lst:
            kvd = _DictKeyAndValue(d)
            self.unjellyInto(kvd, 0, k)
            self.unjellyInto(kvd, 1, v)
        return d

    def _unjelly_module(self, rest):
        moduleName = nativeString(rest[0])
        if type(moduleName) != str:
            raise InsecureJelly('Attempted to unjelly a module with a non-string name.')
        if not self.taster.isModuleAllowed(moduleName):
            raise InsecureJelly(f'Attempted to unjelly module named {moduleName!r}')
        mod = __import__(moduleName, {}, {}, 'x')
        return mod

    def _unjelly_class(self, rest):
        cname = nativeString(rest[0])
        clist = cname.split(nativeString('.'))
        modName = nativeString('.').join(clist[:-1])
        if not self.taster.isModuleAllowed(modName):
            raise InsecureJelly('module %s not allowed' % modName)
        klaus = namedObject(cname)
        objType = type(klaus)
        if objType is not type:
            raise InsecureJelly("class %r unjellied to something that isn't a class: %r" % (cname, klaus))
        if not self.taster.isClassAllowed(klaus):
            raise InsecureJelly('class not allowed: %s' % qual(klaus))
        return klaus

    def _unjelly_function(self, rest):
        fname = nativeString(rest[0])
        modSplit = fname.split(nativeString('.'))
        modName = nativeString('.').join(modSplit[:-1])
        if not self.taster.isModuleAllowed(modName):
            raise InsecureJelly('Module not allowed: %s' % modName)
        function = namedAny(fname)
        return function

    def _unjelly_persistent(self, rest):
        if self.persistentLoad:
            pload = self.persistentLoad(rest[0], self)
            return pload
        else:
            return Unpersistable('Persistent callback not found')

    def _unjelly_instance(self, rest):
        """
        (internal) Unjelly an instance.

        Called to handle the deprecated I{instance} token.

        @param rest: The s-expression representing the instance.

        @return: The unjellied instance.
        """
        warnings.warn_explicit('Unjelly support for the instance atom is deprecated since Twisted 15.0.0.  Upgrade peer for modern instance support.', category=DeprecationWarning, filename='', lineno=0)
        clz = self.unjelly(rest[0])
        return self._genericUnjelly(clz, rest[1])

    def _unjelly_unpersistable(self, rest):
        return Unpersistable(f'Unpersistable data: {rest[0]}')

    def _unjelly_method(self, rest):
        """
        (internal) Unjelly a method.
        """
        im_name = rest[0]
        im_self = self.unjelly(rest[1])
        im_class = self.unjelly(rest[2])
        if not isinstance(im_class, type):
            raise InsecureJelly('Method found with non-class class.')
        if im_name in im_class.__dict__:
            if im_self is None:
                im = getattr(im_class, im_name)
            elif isinstance(im_self, NotKnown):
                im = _InstanceMethod(im_name, im_self, im_class)
            else:
                im = types.MethodType(im_class.__dict__[im_name], im_self, *[im_class] * False)
        else:
            raise TypeError('instance method changed')
        return im