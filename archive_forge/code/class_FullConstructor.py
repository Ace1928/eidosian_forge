from .error import *
from .nodes import *
import collections.abc, datetime, base64, binascii, re, sys, types
class FullConstructor(SafeConstructor):

    def get_state_keys_blacklist(self):
        return ['^extend$', '^__.*__$']

    def get_state_keys_blacklist_regexp(self):
        if not hasattr(self, 'state_keys_blacklist_regexp'):
            self.state_keys_blacklist_regexp = re.compile('(' + '|'.join(self.get_state_keys_blacklist()) + ')')
        return self.state_keys_blacklist_regexp

    def construct_python_str(self, node):
        return self.construct_scalar(node)

    def construct_python_unicode(self, node):
        return self.construct_scalar(node)

    def construct_python_bytes(self, node):
        try:
            value = self.construct_scalar(node).encode('ascii')
        except UnicodeEncodeError as exc:
            raise ConstructorError(None, None, 'failed to convert base64 data into ascii: %s' % exc, node.start_mark)
        try:
            if hasattr(base64, 'decodebytes'):
                return base64.decodebytes(value)
            else:
                return base64.decodestring(value)
        except binascii.Error as exc:
            raise ConstructorError(None, None, 'failed to decode base64 data: %s' % exc, node.start_mark)

    def construct_python_long(self, node):
        return self.construct_yaml_int(node)

    def construct_python_complex(self, node):
        return complex(self.construct_scalar(node))

    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

    def find_python_module(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError('while constructing a Python module', mark, 'expected non-empty name appended to the tag', mark)
        if unsafe:
            try:
                __import__(name)
            except ImportError as exc:
                raise ConstructorError('while constructing a Python module', mark, 'cannot find module %r (%s)' % (name, exc), mark)
        if name not in sys.modules:
            raise ConstructorError('while constructing a Python module', mark, 'module %r is not imported' % name, mark)
        return sys.modules[name]

    def find_python_name(self, name, mark, unsafe=False):
        if not name:
            raise ConstructorError('while constructing a Python object', mark, 'expected non-empty name appended to the tag', mark)
        if '.' in name:
            module_name, object_name = name.rsplit('.', 1)
        else:
            module_name = 'builtins'
            object_name = name
        if unsafe:
            try:
                __import__(module_name)
            except ImportError as exc:
                raise ConstructorError('while constructing a Python object', mark, 'cannot find module %r (%s)' % (module_name, exc), mark)
        if module_name not in sys.modules:
            raise ConstructorError('while constructing a Python object', mark, 'module %r is not imported' % module_name, mark)
        module = sys.modules[module_name]
        if not hasattr(module, object_name):
            raise ConstructorError('while constructing a Python object', mark, 'cannot find %r in the module %r' % (object_name, module.__name__), mark)
        return getattr(module, object_name)

    def construct_python_name(self, suffix, node):
        value = self.construct_scalar(node)
        if value:
            raise ConstructorError('while constructing a Python name', node.start_mark, 'expected the empty value, but found %r' % value, node.start_mark)
        return self.find_python_name(suffix, node.start_mark)

    def construct_python_module(self, suffix, node):
        value = self.construct_scalar(node)
        if value:
            raise ConstructorError('while constructing a Python module', node.start_mark, 'expected the empty value, but found %r' % value, node.start_mark)
        return self.find_python_module(suffix, node.start_mark)

    def make_python_instance(self, suffix, node, args=None, kwds=None, newobj=False, unsafe=False):
        if not args:
            args = []
        if not kwds:
            kwds = {}
        cls = self.find_python_name(suffix, node.start_mark)
        if not (unsafe or isinstance(cls, type)):
            raise ConstructorError('while constructing a Python instance', node.start_mark, 'expected a class, but found %r' % type(cls), node.start_mark)
        if newobj and isinstance(cls, type):
            return cls.__new__(cls, *args, **kwds)
        else:
            return cls(*args, **kwds)

    def set_python_instance_state(self, instance, state, unsafe=False):
        if hasattr(instance, '__setstate__'):
            instance.__setstate__(state)
        else:
            slotstate = {}
            if isinstance(state, tuple) and len(state) == 2:
                state, slotstate = state
            if hasattr(instance, '__dict__'):
                if not unsafe and state:
                    for key in state.keys():
                        self.check_state_key(key)
                instance.__dict__.update(state)
            elif state:
                slotstate.update(state)
            for key, value in slotstate.items():
                if not unsafe:
                    self.check_state_key(key)
                setattr(instance, key, value)

    def construct_python_object(self, suffix, node):
        instance = self.make_python_instance(suffix, node, newobj=True)
        yield instance
        deep = hasattr(instance, '__setstate__')
        state = self.construct_mapping(node, deep=deep)
        self.set_python_instance_state(instance, state)

    def construct_python_object_apply(self, suffix, node, newobj=False):
        if isinstance(node, SequenceNode):
            args = self.construct_sequence(node, deep=True)
            kwds = {}
            state = {}
            listitems = []
            dictitems = {}
        else:
            value = self.construct_mapping(node, deep=True)
            args = value.get('args', [])
            kwds = value.get('kwds', {})
            state = value.get('state', {})
            listitems = value.get('listitems', [])
            dictitems = value.get('dictitems', {})
        instance = self.make_python_instance(suffix, node, args, kwds, newobj)
        if state:
            self.set_python_instance_state(instance, state)
        if listitems:
            instance.extend(listitems)
        if dictitems:
            for key in dictitems:
                instance[key] = dictitems[key]
        return instance

    def construct_python_object_new(self, suffix, node):
        return self.construct_python_object_apply(suffix, node, newobj=True)