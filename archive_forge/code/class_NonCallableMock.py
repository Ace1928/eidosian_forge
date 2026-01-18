from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
class NonCallableMock(Base):
    """A non-callable version of `Mock`"""

    def __new__(cls, *args, **kw):
        new = type(cls.__name__, (cls,), {'__doc__': cls.__doc__})
        instance = object.__new__(new)
        return instance

    def __init__(self, spec=None, wraps=None, name=None, spec_set=None, parent=None, _spec_state=None, _new_name='', _new_parent=None, _spec_as_instance=False, _eat_self=None, unsafe=False, **kwargs):
        if _new_parent is None:
            _new_parent = parent
        __dict__ = self.__dict__
        __dict__['_mock_parent'] = parent
        __dict__['_mock_name'] = name
        __dict__['_mock_new_name'] = _new_name
        __dict__['_mock_new_parent'] = _new_parent
        if spec_set is not None:
            spec = spec_set
            spec_set = True
        if _eat_self is None:
            _eat_self = parent is not None
        self._mock_add_spec(spec, spec_set, _spec_as_instance, _eat_self)
        __dict__['_mock_children'] = {}
        __dict__['_mock_wraps'] = wraps
        __dict__['_mock_delegate'] = None
        __dict__['_mock_called'] = False
        __dict__['_mock_call_args'] = None
        __dict__['_mock_call_count'] = 0
        __dict__['_mock_call_args_list'] = _CallList()
        __dict__['_mock_mock_calls'] = _CallList()
        __dict__['method_calls'] = _CallList()
        __dict__['_mock_unsafe'] = unsafe
        if kwargs:
            self.configure_mock(**kwargs)
        _safe_super(NonCallableMock, self).__init__(spec, wraps, name, spec_set, parent, _spec_state)

    def attach_mock(self, mock, attribute):
        """
        Attach a mock as an attribute of this one, replacing its name and
        parent. Calls to the attached mock will be recorded in the
        `method_calls` and `mock_calls` attributes of this one."""
        mock._mock_parent = None
        mock._mock_new_parent = None
        mock._mock_name = ''
        mock._mock_new_name = None
        setattr(self, attribute, mock)

    def mock_add_spec(self, spec, spec_set=False):
        """Add a spec to a mock. `spec` can either be an object or a
        list of strings. Only attributes on the `spec` can be fetched as
        attributes from the mock.

        If `spec_set` is True then only attributes on the spec can be set."""
        self._mock_add_spec(spec, spec_set)

    def _mock_add_spec(self, spec, spec_set, _spec_as_instance=False, _eat_self=False):
        _spec_class = None
        _spec_signature = None
        if spec is not None and (not _is_list(spec)):
            if isinstance(spec, ClassTypes):
                _spec_class = spec
            else:
                _spec_class = _get_class(spec)
            res = _get_signature_object(spec, _spec_as_instance, _eat_self)
            _spec_signature = res and res[1]
            spec = dir(spec)
        __dict__ = self.__dict__
        __dict__['_spec_class'] = _spec_class
        __dict__['_spec_set'] = spec_set
        __dict__['_spec_signature'] = _spec_signature
        __dict__['_mock_methods'] = spec

    def __get_return_value(self):
        ret = self._mock_return_value
        if self._mock_delegate is not None:
            ret = self._mock_delegate.return_value
        if ret is DEFAULT:
            ret = self._get_child_mock(_new_parent=self, _new_name='()')
            self.return_value = ret
        return ret

    def __set_return_value(self, value):
        if self._mock_delegate is not None:
            self._mock_delegate.return_value = value
        else:
            self._mock_return_value = value
            _check_and_set_parent(self, value, None, '()')
    __return_value_doc = 'The value to be returned when the mock is called.'
    return_value = property(__get_return_value, __set_return_value, __return_value_doc)

    @property
    def __class__(self):
        if self._spec_class is None:
            return type(self)
        return self._spec_class
    called = _delegating_property('called')
    call_count = _delegating_property('call_count')
    call_args = _delegating_property('call_args')
    call_args_list = _delegating_property('call_args_list')
    mock_calls = _delegating_property('mock_calls')

    def __get_side_effect(self):
        delegated = self._mock_delegate
        if delegated is None:
            return self._mock_side_effect
        sf = delegated.side_effect
        if sf is not None and (not callable(sf)) and (not isinstance(sf, _MockIter)) and (not _is_exception(sf)):
            sf = _MockIter(sf)
            delegated.side_effect = sf
        return sf

    def __set_side_effect(self, value):
        value = _try_iter(value)
        delegated = self._mock_delegate
        if delegated is None:
            self._mock_side_effect = value
        else:
            delegated.side_effect = value
    side_effect = property(__get_side_effect, __set_side_effect)

    def reset_mock(self, visited=None):
        """Restore the mock object to its initial state."""
        if visited is None:
            visited = []
        if id(self) in visited:
            return
        visited.append(id(self))
        self.called = False
        self.call_args = None
        self.call_count = 0
        self.mock_calls = _CallList()
        self.call_args_list = _CallList()
        self.method_calls = _CallList()
        for child in self._mock_children.values():
            if isinstance(child, _SpecState):
                continue
            child.reset_mock(visited)
        ret = self._mock_return_value
        if _is_instance_mock(ret) and ret is not self:
            ret.reset_mock(visited)

    def configure_mock(self, **kwargs):
        """Set attributes on the mock through keyword arguments.

        Attributes plus return values and side effects can be set on child
        mocks using standard dot notation and unpacking a dictionary in the
        method call:

        >>> attrs = {'method.return_value': 3, 'other.side_effect': KeyError}
        >>> mock.configure_mock(**attrs)"""
        for arg, val in sorted(kwargs.items(), key=lambda entry: entry[0].count('.')):
            args = arg.split('.')
            final = args.pop()
            obj = self
            for entry in args:
                obj = getattr(obj, entry)
            setattr(obj, final, val)

    def __getattr__(self, name):
        if name in ('_mock_methods', '_mock_unsafe'):
            raise AttributeError(name)
        elif self._mock_methods is not None:
            if name not in self._mock_methods or name in _all_magics:
                raise AttributeError('Mock object has no attribute %r' % name)
        elif _is_magic(name):
            raise AttributeError(name)
        if not self._mock_unsafe:
            if name.startswith(('assert', 'assret')):
                raise AttributeError(name)
        result = self._mock_children.get(name)
        if result is _deleted:
            raise AttributeError(name)
        elif result is None:
            wraps = None
            if self._mock_wraps is not None:
                wraps = getattr(self._mock_wraps, name)
            result = self._get_child_mock(parent=self, name=name, wraps=wraps, _new_name=name, _new_parent=self)
            self._mock_children[name] = result
        elif isinstance(result, _SpecState):
            result = create_autospec(result.spec, result.spec_set, result.instance, result.parent, result.name)
            self._mock_children[name] = result
        return result

    def __repr__(self):
        _name_list = [self._mock_new_name]
        _parent = self._mock_new_parent
        last = self
        dot = '.'
        if _name_list == ['()']:
            dot = ''
        seen = set()
        while _parent is not None:
            last = _parent
            _name_list.append(_parent._mock_new_name + dot)
            dot = '.'
            if _parent._mock_new_name == '()':
                dot = ''
            _parent = _parent._mock_new_parent
            if id(_parent) in seen:
                break
            seen.add(id(_parent))
        _name_list = list(reversed(_name_list))
        _first = last._mock_name or 'mock'
        if len(_name_list) > 1:
            if _name_list[1] not in ('()', '().'):
                _first += '.'
        _name_list[0] = _first
        name = ''.join(_name_list)
        name_string = ''
        if name not in ('mock', 'mock.'):
            name_string = ' name=%r' % name
        spec_string = ''
        if self._spec_class is not None:
            spec_string = ' spec=%r'
            if self._spec_set:
                spec_string = ' spec_set=%r'
            spec_string = spec_string % self._spec_class.__name__
        return "<%s%s%s id='%s'>" % (type(self).__name__, name_string, spec_string, id(self))

    def __dir__(self):
        """Filter the output of `dir(mock)` to only useful members."""
        if not mock.FILTER_DIR and getattr(object, '__dir__', None):
            return object.__dir__(self)
        extras = self._mock_methods or []
        from_type = dir(type(self))
        from_dict = list(self.__dict__)
        if mock.FILTER_DIR:
            from_type = [e for e in from_type if not e.startswith('_')]
            from_dict = [e for e in from_dict if not e.startswith('_') or _is_magic(e)]
        return sorted(set(extras + from_type + from_dict + list(self._mock_children)))

    def __setattr__(self, name, value):
        if name in _allowed_names:
            return object.__setattr__(self, name, value)
        elif self._spec_set and self._mock_methods is not None and (name not in self._mock_methods) and (name not in self.__dict__):
            raise AttributeError("Mock object has no attribute '%s'" % name)
        elif name in _unsupported_magics:
            msg = 'Attempting to set unsupported magic method %r.' % name
            raise AttributeError(msg)
        elif name in _all_magics:
            if self._mock_methods is not None and name not in self._mock_methods:
                raise AttributeError("Mock object has no attribute '%s'" % name)
            if not _is_instance_mock(value):
                setattr(type(self), name, _get_method(name, value))
                original = value
                value = lambda *args, **kw: original(self, *args, **kw)
            else:
                _check_and_set_parent(self, value, None, name)
                setattr(type(self), name, value)
                self._mock_children[name] = value
        elif name == '__class__':
            self._spec_class = value
            return
        elif _check_and_set_parent(self, value, name, name):
            self._mock_children[name] = value
        return object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in _all_magics and name in type(self).__dict__:
            delattr(type(self), name)
            if name not in self.__dict__:
                return
        if name in self.__dict__:
            object.__delattr__(self, name)
        obj = self._mock_children.get(name, _missing)
        if obj is _deleted:
            raise AttributeError(name)
        if obj is not _missing:
            del self._mock_children[name]
        self._mock_children[name] = _deleted

    def _format_mock_call_signature(self, args, kwargs):
        name = self._mock_name or 'mock'
        return _format_call_signature(name, args, kwargs)

    def _format_mock_failure_message(self, args, kwargs):
        message = 'Expected call: %s\nActual call: %s'
        expected_string = self._format_mock_call_signature(args, kwargs)
        call_args = self.call_args
        if len(call_args) == 3:
            call_args = call_args[1:]
        actual_string = self._format_mock_call_signature(*call_args)
        return message % (expected_string, actual_string)

    def _call_matcher(self, _call):
        """
        Given a call (or simply a (args, kwargs) tuple), return a
        comparison key suitable for matching with other calls.
        This is a best effort method which relies on the spec's signature,
        if available, or falls back on the arguments themselves.
        """
        sig = self._spec_signature
        if sig is not None:
            if len(_call) == 2:
                name = ''
                args, kwargs = _call
            else:
                name, args, kwargs = _call
            try:
                return (name, sig.bind(*args, **kwargs))
            except TypeError as e:
                e.__traceback__ = None
                return e
        else:
            return _call

    def assert_not_called(_mock_self):
        """assert that the mock was never called.
        """
        self = _mock_self
        if self.call_count != 0:
            msg = "Expected '%s' to not have been called. Called %s times." % (self._mock_name or 'mock', self.call_count)
            raise AssertionError(msg)

    def assert_called(_mock_self):
        """assert that the mock was called at least once
        """
        self = _mock_self
        if self.call_count == 0:
            msg = "Expected '%s' to have been called." % self._mock_name or 'mock'
            raise AssertionError(msg)

    def assert_called_once(_mock_self):
        """assert that the mock was called only once.
        """
        self = _mock_self
        if not self.call_count == 1:
            msg = "Expected '%s' to have been called once. Called %s times." % (self._mock_name or 'mock', self.call_count)
            raise AssertionError(msg)

    def assert_called_with(_mock_self, *args, **kwargs):
        """assert that the mock was called with the specified arguments.

        Raises an AssertionError if the args and keyword args passed in are
        different to the last call to the mock."""
        self = _mock_self
        if self.call_args is None:
            expected = self._format_mock_call_signature(args, kwargs)
            raise AssertionError('Expected call: %s\nNot called' % (expected,))

        def _error_message(cause):
            msg = self._format_mock_failure_message(args, kwargs)
            if six.PY2 and cause is not None:
                msg = '%s\n%s' % (msg, str(cause))
            return msg
        expected = self._call_matcher((args, kwargs))
        actual = self._call_matcher(self.call_args)
        if expected != actual:
            cause = expected if isinstance(expected, Exception) else None
            six.raise_from(AssertionError(_error_message(cause)), cause)

    def assert_called_once_with(_mock_self, *args, **kwargs):
        """assert that the mock was called exactly once and with the specified
        arguments."""
        self = _mock_self
        if not self.call_count == 1:
            msg = "Expected '%s' to be called once. Called %s times." % (self._mock_name or 'mock', self.call_count)
            raise AssertionError(msg)
        return self.assert_called_with(*args, **kwargs)

    def assert_has_calls(self, calls, any_order=False):
        """assert the mock has been called with the specified calls.
        The `mock_calls` list is checked for the calls.

        If `any_order` is False (the default) then the calls must be
        sequential. There can be extra calls before or after the
        specified calls.

        If `any_order` is True then the calls can be in any order, but
        they must all appear in `mock_calls`."""
        expected = [self._call_matcher(c) for c in calls]
        cause = expected if isinstance(expected, Exception) else None
        all_calls = _CallList((self._call_matcher(c) for c in self.mock_calls))
        if not any_order:
            if expected not in all_calls:
                six.raise_from(AssertionError('Calls not found.\nExpected: %r\nActual: %r' % (_CallList(calls), self.mock_calls)), cause)
            return
        all_calls = list(all_calls)
        not_found = []
        for kall in expected:
            try:
                all_calls.remove(kall)
            except ValueError:
                not_found.append(kall)
        if not_found:
            six.raise_from(AssertionError('%r not all found in call list' % (tuple(not_found),)), cause)

    def assert_any_call(self, *args, **kwargs):
        """assert the mock has been called with the specified arguments.

        The assert passes if the mock has *ever* been called, unlike
        `assert_called_with` and `assert_called_once_with` that only pass if
        the call is the most recent one."""
        expected = self._call_matcher((args, kwargs))
        actual = [self._call_matcher(c) for c in self.call_args_list]
        if expected not in actual:
            cause = expected if isinstance(expected, Exception) else None
            expected_string = self._format_mock_call_signature(args, kwargs)
            six.raise_from(AssertionError('%s call not found' % expected_string), cause)

    def _get_child_mock(self, **kw):
        """Create the child mocks for attributes and return value.
        By default child mocks will be the same type as the parent.
        Subclasses of Mock may want to override this to customize the way
        child mocks are made.

        For non-callable mocks the callable variant will be used (rather than
        any custom subclass)."""
        _type = type(self)
        if not issubclass(_type, CallableMixin):
            if issubclass(_type, NonCallableMagicMock):
                klass = MagicMock
            elif issubclass(_type, NonCallableMock):
                klass = Mock
        else:
            klass = _type.__mro__[1]
        return klass(**kw)