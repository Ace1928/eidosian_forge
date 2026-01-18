import copyreg as copy_reg
import re
import types
from twisted.persisted import crefutil
from twisted.python import log, reflect
from twisted.python.compat import _constructMethod
from ._tokenize import generate_tokens as tokenize
class AOTJellier:

    def __init__(self):
        self.prepared = {}
        self._ref_id = 0
        self.stack = []

    def prepareForRef(self, aoref, object):
        """I prepare an object for later referencing, by storing its id()
        and its _AORef in a cache."""
        self.prepared[id(object)] = aoref

    def jellyToAO(self, obj):
        """I turn an object into an AOT and return it."""
        objType = type(obj)
        self.stack.append(repr(obj))
        if objType in _SIMPLE_BUILTINS:
            retval = obj
        elif issubclass(objType, types.MethodType):
            retval = InstanceMethod(_funcOfMethod(obj).__name__, reflect.qual(_classOfMethod(obj)), self.jellyToAO(_selfOfMethod(obj)))
        elif issubclass(objType, types.ModuleType):
            retval = Module(obj.__name__)
        elif issubclass(objType, type):
            retval = Class(reflect.qual(obj))
        elif objType is types.FunctionType:
            retval = Function(reflect.fullFuncName(obj))
        else:
            if id(obj) in self.prepared:
                oldRef = self.prepared[id(obj)]
                if oldRef.refnum:
                    key = oldRef.refnum
                else:
                    self._ref_id = self._ref_id + 1
                    key = self._ref_id
                    oldRef.setRef(key)
                return Deref(key)
            retval = Ref()

            def _stateFrom(state):
                retval.setObj(Instance(reflect.qual(obj.__class__), self.jellyToAO(state)))
            self.prepareForRef(retval, obj)
            if objType is list:
                retval.setObj([self.jellyToAO(o) for o in obj])
            elif objType is tuple:
                retval.setObj(tuple(map(self.jellyToAO, obj)))
            elif objType is dict:
                d = {}
                for k, v in obj.items():
                    d[self.jellyToAO(k)] = self.jellyToAO(v)
                retval.setObj(d)
            elif objType in copy_reg.dispatch_table:
                unpickleFunc, state = copy_reg.dispatch_table[objType](obj)
                retval.setObj(Copyreg(reflect.fullFuncName(unpickleFunc), self.jellyToAO(state)))
            elif hasattr(obj, '__getstate__'):
                _stateFrom(obj.__getstate__())
            elif hasattr(obj, '__dict__'):
                _stateFrom(obj.__dict__)
            else:
                raise TypeError('Unsupported type: %s' % objType.__name__)
        del self.stack[-1]
        return retval

    def jelly(self, obj):
        try:
            ao = self.jellyToAO(obj)
            return ao
        except BaseException:
            log.msg('Error jellying object! Stacktrace follows::')
            log.msg('\n'.join(self.stack))
            raise