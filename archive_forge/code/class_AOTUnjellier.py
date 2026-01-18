import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
class AOTUnjellier:
    """I handle the unjellying of an Abstract Object Tree.
    See AOTUnjellier.unjellyAO
    """

    def __init__(self):
        self.references = {}
        self.stack = []
        self.afterUnjelly = []

    def unjellyLater(self, node):
        """Unjelly a node, later."""
        d = crefutil._Defer()
        self.unjellyInto(d, 0, node)
        return d

    def unjellyInto(self, obj, loc, ao):
        """Utility method for unjellying one object into another.
        This automates the handling of backreferences.
        """
        o = self.unjellyAO(ao)
        obj[loc] = o
        if isinstance(o, crefutil.NotKnown):
            o.addDependant(obj, loc)
        return o

    def callAfter(self, callable, result):
        if isinstance(result, crefutil.NotKnown):
            listResult = [None]
            result.addDependant(listResult, 1)
        else:
            listResult = [result]
        self.afterUnjelly.append((callable, listResult))

    def unjellyAttribute(self, instance, attrName, ao):
        """Utility method for unjellying into instances of attributes.

        Use this rather than unjellyAO unless you like surprising bugs!
        Alternatively, you can use unjellyInto on your instance's __dict__.
        """
        self.unjellyInto(instance.__dict__, attrName, ao)

    def unjellyAO(self, ao):
        """Unjelly an Abstract Object and everything it contains.
        I return the real object.
        """
        self.stack.append(ao)
        t = type(ao)
        if t in _SIMPLE_BUILTINS:
            return ao
        elif t is list:
            l = []
            for x in ao:
                l.append(None)
                self.unjellyInto(l, len(l) - 1, x)
            return l
        elif t is tuple:
            l = []
            tuple_ = tuple
            for x in ao:
                l.append(None)
                if isinstance(self.unjellyInto(l, len(l) - 1, x), crefutil.NotKnown):
                    tuple_ = crefutil._Tuple
            return tuple_(l)
        elif t is dict:
            d = {}
            for k, v in ao.items():
                kvd = crefutil._DictKeyAndValue(d)
                self.unjellyInto(kvd, 0, k)
                self.unjellyInto(kvd, 1, v)
            return d
        else:
            c = ao.__class__
            if c is Module:
                return reflect.namedModule(ao.name)
            elif c in [Class, Function] or issubclass(c, type):
                return reflect.namedObject(ao.name)
            elif c is InstanceMethod:
                im_name = ao.name
                im_class = reflect.namedObject(ao.klass)
                im_self = self.unjellyAO(ao.instance)
                if im_name in im_class.__dict__:
                    if im_self is None:
                        return getattr(im_class, im_name)
                    elif isinstance(im_self, crefutil.NotKnown):
                        return crefutil._InstanceMethod(im_name, im_self, im_class)
                    else:
                        return _constructMethod(im_class, im_name, im_self)
                else:
                    raise TypeError('instance method changed')
            elif c is Instance:
                klass = reflect.namedObject(ao.klass)
                state = self.unjellyAO(ao.state)
                inst = klass.__new__(klass)
                if hasattr(klass, '__setstate__'):
                    self.callAfter(inst.__setstate__, state)
                elif isinstance(state, dict):
                    inst.__dict__ = state
                else:
                    inst.__dict__ = state.__getstate__()
                return inst
            elif c is Ref:
                o = self.unjellyAO(ao.obj)
                refkey = ao.refnum
                ref = self.references.get(refkey)
                if ref is None:
                    self.references[refkey] = o
                elif isinstance(ref, crefutil.NotKnown):
                    ref.resolveDependants(o)
                    self.references[refkey] = o
                elif refkey is None:
                    pass
                else:
                    raise ValueError('Multiple references with the same ID: %s, %s, %s!' % (ref, refkey, ao))
                return o
            elif c is Deref:
                num = ao.refnum
                ref = self.references.get(num)
                if ref is None:
                    der = crefutil._Dereference(num)
                    self.references[num] = der
                    return der
                return ref
            elif c is Copyreg:
                loadfunc = reflect.namedObject(ao.loadfunc)
                d = self.unjellyLater(ao.state).addCallback(lambda result, _l: _l(*result), loadfunc)
                return d
            else:
                raise TypeError('Unsupported AOT type: %s' % t)

    def unjelly(self, ao):
        try:
            l = [None]
            self.unjellyInto(l, 0, ao)
            for func, v in self.afterUnjelly:
                func(v[0])
            return l[0]
        except BaseException:
            log.msg('Error jellying object! Stacktrace follows::')
            log.msg('\n'.join(map(repr, self.stack)))
            raise