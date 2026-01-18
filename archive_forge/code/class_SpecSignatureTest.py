import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
class SpecSignatureTest(unittest.TestCase):

    def _check_someclass_mock(self, mock):
        self.assertRaises(AttributeError, getattr, mock, 'foo')
        mock.one(1, 2)
        mock.one.assert_called_with(1, 2)
        self.assertRaises(AssertionError, mock.one.assert_called_with, 3, 4)
        self.assertRaises(TypeError, mock.one, 1)
        mock.two()
        mock.two.assert_called_with()
        self.assertRaises(AssertionError, mock.two.assert_called_with, 3)
        self.assertRaises(TypeError, mock.two, 1)
        mock.three()
        mock.three.assert_called_with()
        self.assertRaises(AssertionError, mock.three.assert_called_with, 3)
        self.assertRaises(TypeError, mock.three, 3, 2)
        mock.three(1)
        mock.three.assert_called_with(1)
        mock.three(a=1)
        mock.three.assert_called_with(a=1)

    def test_basic(self):
        mock = create_autospec(SomeClass)
        self._check_someclass_mock(mock)
        mock = create_autospec(SomeClass())
        self._check_someclass_mock(mock)

    def test_create_autospec_return_value(self):

        def f():
            pass
        mock = create_autospec(f, return_value='foo')
        self.assertEqual(mock(), 'foo')

        class Foo(object):
            pass
        mock = create_autospec(Foo, return_value='foo')
        self.assertEqual(mock(), 'foo')

    def test_autospec_reset_mock(self):
        m = create_autospec(int)
        int(m)
        m.reset_mock()
        self.assertEqual(m.__int__.call_count, 0)

    def test_mocking_unbound_methods(self):

        class Foo(object):

            def foo(self, foo):
                pass
        p = patch.object(Foo, 'foo')
        mock_foo = p.start()
        Foo().foo(1)
        mock_foo.assert_called_with(1)

    @unittest.expectedFailure
    def test_create_autospec_unbound_methods(self):

        class Foo(object):

            def foo(self):
                pass
        klass = create_autospec(Foo)
        instance = klass()
        self.assertRaises(TypeError, instance.foo, 1)
        klass.foo(1)
        klass.foo.assert_called_with(1)
        self.assertRaises(TypeError, klass.foo)

    def test_create_autospec_keyword_arguments(self):

        class Foo(object):
            a = 3
        m = create_autospec(Foo, a='3')
        self.assertEqual(m.a, '3')

    @unittest.skipUnless(six.PY3, 'Keyword only arguments Python 3 specific')
    def test_create_autospec_keyword_only_arguments(self):
        func_def = 'def foo(a, *, b=None):\n    pass\n'
        namespace = {}
        exec(func_def, namespace)
        foo = namespace['foo']
        m = create_autospec(foo)
        m(1)
        m.assert_called_with(1)
        self.assertRaises(TypeError, m, 1, 2)
        m(2, b=3)
        m.assert_called_with(2, b=3)

    def test_function_as_instance_attribute(self):
        obj = SomeClass()

        def f(a):
            pass
        obj.f = f
        mock = create_autospec(obj)
        mock.f('bing')
        mock.f.assert_called_with('bing')

    def test_spec_as_list(self):
        mock = create_autospec([])
        mock.append('foo')
        mock.append.assert_called_with('foo')
        self.assertRaises(AttributeError, getattr, mock, 'foo')

        class Foo(object):
            foo = []
        mock = create_autospec(Foo)
        mock.foo.append(3)
        mock.foo.append.assert_called_with(3)
        self.assertRaises(AttributeError, getattr, mock.foo, 'foo')

    def test_attributes(self):

        class Sub(SomeClass):
            attr = SomeClass()
        sub_mock = create_autospec(Sub)
        for mock in (sub_mock, sub_mock.attr):
            self._check_someclass_mock(mock)

    def test_builtin_functions_types(self):

        class BuiltinSubclass(list):

            def bar(self, arg):
                pass
            sorted = sorted
            attr = {}
        mock = create_autospec(BuiltinSubclass)
        mock.append(3)
        mock.append.assert_called_with(3)
        self.assertRaises(AttributeError, getattr, mock.append, 'foo')
        mock.bar('foo')
        mock.bar.assert_called_with('foo')
        self.assertRaises(TypeError, mock.bar, 'foo', 'bar')
        self.assertRaises(AttributeError, getattr, mock.bar, 'foo')
        mock.sorted([1, 2])
        mock.sorted.assert_called_with([1, 2])
        self.assertRaises(AttributeError, getattr, mock.sorted, 'foo')
        mock.attr.pop(3)
        mock.attr.pop.assert_called_with(3)
        self.assertRaises(AttributeError, getattr, mock.attr, 'foo')

    def test_method_calls(self):

        class Sub(SomeClass):
            attr = SomeClass()
        mock = create_autospec(Sub)
        mock.one(1, 2)
        mock.two()
        mock.three(3)
        expected = [call.one(1, 2), call.two(), call.three(3)]
        self.assertEqual(mock.method_calls, expected)
        mock.attr.one(1, 2)
        mock.attr.two()
        mock.attr.three(3)
        expected.extend([call.attr.one(1, 2), call.attr.two(), call.attr.three(3)])
        self.assertEqual(mock.method_calls, expected)

    def test_magic_methods(self):

        class BuiltinSubclass(list):
            attr = {}
        mock = create_autospec(BuiltinSubclass)
        self.assertEqual(list(mock), [])
        self.assertRaises(TypeError, int, mock)
        self.assertRaises(TypeError, int, mock.attr)
        self.assertEqual(list(mock), [])
        self.assertIsInstance(mock['foo'], MagicMock)
        self.assertIsInstance(mock.attr['foo'], MagicMock)

    def test_spec_set(self):

        class Sub(SomeClass):
            attr = SomeClass()
        for spec in (Sub, Sub()):
            mock = create_autospec(spec, spec_set=True)
            self._check_someclass_mock(mock)
            self.assertRaises(AttributeError, setattr, mock, 'foo', 'bar')
            self.assertRaises(AttributeError, setattr, mock.attr, 'foo', 'bar')

    def test_descriptors(self):

        class Foo(object):

            @classmethod
            def f(cls, a, b):
                pass

            @staticmethod
            def g(a, b):
                pass

        class Bar(Foo):
            pass

        class Baz(SomeClass, Bar):
            pass
        for spec in (Foo, Foo(), Bar, Bar(), Baz, Baz()):
            mock = create_autospec(spec)
            mock.f(1, 2)
            mock.f.assert_called_once_with(1, 2)
            mock.g(3, 4)
            mock.g.assert_called_once_with(3, 4)

    @unittest.skipIf(six.PY3, 'No old style classes in Python 3')
    def test_old_style_classes(self):

        class Foo:

            def f(self, a, b):
                pass

        class Bar(Foo):
            g = Foo()
        for spec in (Foo, Foo(), Bar, Bar()):
            mock = create_autospec(spec)
            mock.f(1, 2)
            mock.f.assert_called_once_with(1, 2)
            self.assertRaises(AttributeError, getattr, mock, 'foo')
            self.assertRaises(AttributeError, getattr, mock.f, 'foo')
        mock.g.f(1, 2)
        mock.g.f.assert_called_once_with(1, 2)
        self.assertRaises(AttributeError, getattr, mock.g, 'foo')

    def test_recursive(self):

        class A(object):

            def a(self):
                pass
            foo = 'foo bar baz'
            bar = foo
        A.B = A
        mock = create_autospec(A)
        mock()
        self.assertFalse(mock.B.called)
        mock.a()
        mock.B.a()
        self.assertEqual(mock.method_calls, [call.a(), call.B.a()])
        self.assertIs(A.foo, A.bar)
        self.assertIsNot(mock.foo, mock.bar)
        mock.foo.lower()
        self.assertRaises(AssertionError, mock.bar.lower.assert_called_with)

    def test_spec_inheritance_for_classes(self):

        class Foo(object):

            def a(self, x):
                pass

            class Bar(object):

                def f(self, y):
                    pass
        class_mock = create_autospec(Foo)
        self.assertIsNot(class_mock, class_mock())
        for this_mock in (class_mock, class_mock()):
            this_mock.a(x=5)
            this_mock.a.assert_called_with(x=5)
            this_mock.a.assert_called_with(5)
            self.assertRaises(TypeError, this_mock.a, 'foo', 'bar')
            self.assertRaises(AttributeError, getattr, this_mock, 'b')
        instance_mock = create_autospec(Foo())
        instance_mock.a(5)
        instance_mock.a.assert_called_with(5)
        instance_mock.a.assert_called_with(x=5)
        self.assertRaises(TypeError, instance_mock.a, 'foo', 'bar')
        self.assertRaises(AttributeError, getattr, instance_mock, 'b')
        self.assertRaises(TypeError, instance_mock)
        instance_mock.Bar.f(6)
        instance_mock.Bar.f.assert_called_with(6)
        instance_mock.Bar.f.assert_called_with(y=6)
        self.assertRaises(AttributeError, getattr, instance_mock.Bar, 'g')
        instance_mock.Bar().f(6)
        instance_mock.Bar().f.assert_called_with(6)
        instance_mock.Bar().f.assert_called_with(y=6)
        self.assertRaises(AttributeError, getattr, instance_mock.Bar(), 'g')

    def test_inherit(self):

        class Foo(object):
            a = 3
        Foo.Foo = Foo
        mock = create_autospec(Foo)
        instance = mock()
        self.assertRaises(AttributeError, getattr, instance, 'b')
        attr_instance = mock.Foo()
        self.assertRaises(AttributeError, getattr, attr_instance, 'b')
        mock = create_autospec(Foo())
        self.assertRaises(AttributeError, getattr, mock, 'b')
        self.assertRaises(TypeError, mock)
        call_result = mock.Foo()
        self.assertRaises(AttributeError, getattr, call_result, 'b')

    def test_builtins(self):
        create_autospec(1)
        create_autospec(int)
        create_autospec('foo')
        create_autospec(str)
        create_autospec({})
        create_autospec(dict)
        create_autospec([])
        create_autospec(list)
        create_autospec(set())
        create_autospec(set)
        create_autospec(1.0)
        create_autospec(float)
        create_autospec(1j)
        create_autospec(complex)
        create_autospec(False)
        create_autospec(True)

    def test_function(self):

        def f(a, b):
            pass
        mock = create_autospec(f)
        self.assertRaises(TypeError, mock)
        mock(1, 2)
        mock.assert_called_with(1, 2)
        mock.assert_called_with(1, b=2)
        mock.assert_called_with(a=1, b=2)
        f.f = f
        mock = create_autospec(f)
        self.assertRaises(TypeError, mock.f)
        mock.f(3, 4)
        mock.f.assert_called_with(3, 4)
        mock.f.assert_called_with(a=3, b=4)

    def test_skip_attributeerrors(self):

        class Raiser(object):

            def __get__(self, obj, type=None):
                if obj is None:
                    raise AttributeError('Can only be accessed via an instance')

        class RaiserClass(object):
            raiser = Raiser()

            @staticmethod
            def existing(a, b):
                return a + b
        s = create_autospec(RaiserClass)
        self.assertRaises(TypeError, lambda x: s.existing(1, 2, 3))
        s.existing(1, 2)
        self.assertRaises(AttributeError, lambda: s.nonexisting)
        obj = s.raiser
        (obj.foo, obj.bar)

    def test_signature_class(self):

        class Foo(object):

            def __init__(self, a, b=3):
                pass
        mock = create_autospec(Foo)
        self.assertRaises(TypeError, mock)
        mock(1)
        mock.assert_called_once_with(1)
        mock(4, 5)
        mock.assert_called_with(4, 5)

    @unittest.skipIf(six.PY3, 'no old style classes in Python 3')
    def test_signature_old_style_class(self):

        class Foo:

            def __init__(self, a, b=3):
                pass
        mock = create_autospec(Foo)
        self.assertRaises(TypeError, mock)
        mock(1)
        mock.assert_called_once_with(1)
        mock.assert_called_once_with(a=1)
        self.assertRaises(AssertionError, mock.assert_called_once_with, 2)
        mock(4, 5)
        mock.assert_called_with(4, 5)
        mock.assert_called_with(a=4, b=5)
        self.assertRaises(AssertionError, mock.assert_called_with, a=5, b=4)

    def test_class_with_no_init(self):

        class Foo(object):
            pass
        create_autospec(Foo)

    @unittest.skipIf(six.PY3, 'no old style classes in Python 3')
    def test_old_style_class_with_no_init(self):

        class Foo:
            pass
        create_autospec(Foo)

    def test_signature_callable(self):

        class Callable(object):

            def __init__(self, x, y):
                pass

            def __call__(self, a):
                pass
        mock = create_autospec(Callable)
        mock(1, 2)
        mock.assert_called_once_with(1, 2)
        mock.assert_called_once_with(x=1, y=2)
        self.assertRaises(TypeError, mock, 'a')
        instance = mock(1, 2)
        self.assertRaises(TypeError, instance)
        instance(a='a')
        instance.assert_called_once_with('a')
        instance.assert_called_once_with(a='a')
        instance('a')
        instance.assert_called_with('a')
        instance.assert_called_with(a='a')
        mock = create_autospec(Callable(1, 2))
        mock(a='a')
        mock.assert_called_once_with(a='a')
        self.assertRaises(TypeError, mock)
        mock('a')
        mock.assert_called_with('a')

    def test_signature_noncallable(self):

        class NonCallable(object):

            def __init__(self):
                pass
        mock = create_autospec(NonCallable)
        instance = mock()
        mock.assert_called_once_with()
        self.assertRaises(TypeError, mock, 'a')
        self.assertRaises(TypeError, instance)
        self.assertRaises(TypeError, instance, 'a')
        mock = create_autospec(NonCallable())
        self.assertRaises(TypeError, mock)
        self.assertRaises(TypeError, mock, 'a')

    def test_create_autospec_none(self):

        class Foo(object):
            bar = None
        mock = create_autospec(Foo)
        none = mock.bar
        self.assertNotIsInstance(none, type(None))
        none.foo()
        none.foo.assert_called_once_with()

    def test_autospec_functions_with_self_in_odd_place(self):

        class Foo(object):

            def f(a, self):
                pass
        a = create_autospec(Foo)
        a.f(10)
        a.f.assert_called_with(10)
        a.f.assert_called_with(self=10)
        a.f(self=10)
        a.f.assert_called_with(10)
        a.f.assert_called_with(self=10)

    def test_autospec_property(self):

        class Foo(object):

            @property
            def foo(self):
                return 3
        foo = create_autospec(Foo)
        mock_property = foo.foo
        self.assertIsInstance(mock_property, MagicMock)
        mock_property(1, 2, 3)
        mock_property.abc(4, 5, 6)
        mock_property.assert_called_once_with(1, 2, 3)
        mock_property.abc.assert_called_once_with(4, 5, 6)

    def test_autospec_slots(self):

        class Foo(object):
            __slots__ = ['a']
        foo = create_autospec(Foo)
        mock_slot = foo.a
        mock_slot(1, 2, 3)
        mock_slot.abc(4, 5, 6)
        mock_slot.assert_called_once_with(1, 2, 3)
        mock_slot.abc.assert_called_once_with(4, 5, 6)