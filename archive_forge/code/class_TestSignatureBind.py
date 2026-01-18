from __future__ import absolute_import, division, print_function
import collections
import functools
import sys
import unittest2 as unittest
import funcsigs as inspect
class TestSignatureBind(unittest.TestCase):

    @staticmethod
    def call(func, *args, **kwargs):
        sig = inspect.signature(func)
        ba = sig.bind(*args, **kwargs)
        return func(*ba.args, **ba.kwargs)

    def test_signature_bind_empty(self):

        def test():
            return 42
        self.assertEqual(self.call(test), 42)
        with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
            self.call(test, 1)
        with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
            self.call(test, 1, spam=10)
        with self.assertRaisesRegex(TypeError, 'too many keyword arguments'):
            self.call(test, spam=1)

    def test_signature_bind_var(self):

        def test(*args, **kwargs):
            return (args, kwargs)
        self.assertEqual(self.call(test), ((), {}))
        self.assertEqual(self.call(test, 1), ((1,), {}))
        self.assertEqual(self.call(test, 1, 2), ((1, 2), {}))
        self.assertEqual(self.call(test, foo='bar'), ((), {'foo': 'bar'}))
        self.assertEqual(self.call(test, 1, foo='bar'), ((1,), {'foo': 'bar'}))
        self.assertEqual(self.call(test, args=10), ((), {'args': 10}))
        self.assertEqual(self.call(test, 1, 2, foo='bar'), ((1, 2), {'foo': 'bar'}))

    def test_signature_bind_just_args(self):

        def test(a, b, c):
            return (a, b, c)
        self.assertEqual(self.call(test, 1, 2, 3), (1, 2, 3))
        with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
            self.call(test, 1, 2, 3, 4)
        with self.assertRaisesRegex(TypeError, "'b' parameter lacking default"):
            self.call(test, 1)
        with self.assertRaisesRegex(TypeError, "'a' parameter lacking default"):
            self.call(test)

        def test(a, b, c=10):
            return (a, b, c)
        self.assertEqual(self.call(test, 1, 2, 3), (1, 2, 3))
        self.assertEqual(self.call(test, 1, 2), (1, 2, 10))

        def test(a=1, b=2, c=3):
            return (a, b, c)
        self.assertEqual(self.call(test, a=10, c=13), (10, 2, 13))
        self.assertEqual(self.call(test, a=10), (10, 2, 3))
        self.assertEqual(self.call(test, b=10), (1, 10, 3))

    def test_signature_bind_varargs_order(self):

        def test(*args):
            return args
        self.assertEqual(self.call(test), ())
        self.assertEqual(self.call(test, 1, 2, 3), (1, 2, 3))

    def test_signature_bind_args_and_varargs(self):

        def test(a, b, c=3, *args):
            return (a, b, c, args)
        self.assertEqual(self.call(test, 1, 2, 3, 4, 5), (1, 2, 3, (4, 5)))
        self.assertEqual(self.call(test, 1, 2), (1, 2, 3, ()))
        self.assertEqual(self.call(test, b=1, a=2), (2, 1, 3, ()))
        self.assertEqual(self.call(test, 1, b=2), (1, 2, 3, ()))
        with self.assertRaisesRegex(TypeError, "multiple values for argument 'c'"):
            self.call(test, 1, 2, 3, c=4)

    def test_signature_bind_just_kwargs(self):

        def test(**kwargs):
            return kwargs
        self.assertEqual(self.call(test), {})
        self.assertEqual(self.call(test, foo='bar', spam='ham'), {'foo': 'bar', 'spam': 'ham'})

    def test_signature_bind_args_and_kwargs(self):

        def test(a, b, c=3, **kwargs):
            return (a, b, c, kwargs)
        self.assertEqual(self.call(test, 1, 2), (1, 2, 3, {}))
        self.assertEqual(self.call(test, 1, 2, foo='bar', spam='ham'), (1, 2, 3, {'foo': 'bar', 'spam': 'ham'}))
        self.assertEqual(self.call(test, b=2, a=1, foo='bar', spam='ham'), (1, 2, 3, {'foo': 'bar', 'spam': 'ham'}))
        self.assertEqual(self.call(test, a=1, b=2, foo='bar', spam='ham'), (1, 2, 3, {'foo': 'bar', 'spam': 'ham'}))
        self.assertEqual(self.call(test, 1, b=2, foo='bar', spam='ham'), (1, 2, 3, {'foo': 'bar', 'spam': 'ham'}))
        self.assertEqual(self.call(test, 1, b=2, c=4, foo='bar', spam='ham'), (1, 2, 4, {'foo': 'bar', 'spam': 'ham'}))
        self.assertEqual(self.call(test, 1, 2, 4, foo='bar'), (1, 2, 4, {'foo': 'bar'}))
        self.assertEqual(self.call(test, c=5, a=4, b=3), (4, 3, 5, {}))
    if sys.version_info[0] > 2:
        exec('\ndef test_signature_bind_kwonly(self):\n    def test(*, foo):\n        return foo\n    with self.assertRaisesRegex(TypeError,\n                                 \'too many positional arguments\'):\n        self.call(test, 1)\n    self.assertEqual(self.call(test, foo=1), 1)\n\n    def test(a, *, foo=1, bar):\n        return foo\n    with self.assertRaisesRegex(TypeError,\n                                 "\'bar\' parameter lacking default value"):\n        self.call(test, 1)\n\n    def test(foo, *, bar):\n        return foo, bar\n    self.assertEqual(self.call(test, 1, bar=2), (1, 2))\n    self.assertEqual(self.call(test, bar=2, foo=1), (1, 2))\n\n    with self.assertRaisesRegex(TypeError,\n                                 \'too many keyword arguments\'):\n        self.call(test, bar=2, foo=1, spam=10)\n\n    with self.assertRaisesRegex(TypeError,\n                                 \'too many positional arguments\'):\n        self.call(test, 1, 2)\n\n    with self.assertRaisesRegex(TypeError,\n                                 \'too many positional arguments\'):\n        self.call(test, 1, 2, bar=2)\n\n    with self.assertRaisesRegex(TypeError,\n                                 \'too many keyword arguments\'):\n        self.call(test, 1, bar=2, spam=\'ham\')\n\n    with self.assertRaisesRegex(TypeError,\n                                 "\'bar\' parameter lacking default value"):\n        self.call(test, 1)\n\n    def test(foo, *, bar, **bin):\n        return foo, bar, bin\n    self.assertEqual(self.call(test, 1, bar=2), (1, 2, {}))\n    self.assertEqual(self.call(test, foo=1, bar=2), (1, 2, {}))\n    self.assertEqual(self.call(test, 1, bar=2, spam=\'ham\'),\n                     (1, 2, {\'spam\': \'ham\'}))\n    self.assertEqual(self.call(test, spam=\'ham\', foo=1, bar=2),\n                     (1, 2, {\'spam\': \'ham\'}))\n    with self.assertRaisesRegex(TypeError,\n                                 "\'foo\' parameter lacking default value"):\n        self.call(test, spam=\'ham\', bar=2)\n    self.assertEqual(self.call(test, 1, bar=2, bin=1, spam=10),\n                     (1, 2, {\'bin\': 1, \'spam\': 10}))\n')
    if sys.version_info[0] > 2:
        exec("\ndef test_signature_bind_arguments(self):\n    def test(a, *args, b, z=100, **kwargs):\n        pass\n    sig = inspect.signature(test)\n    ba = sig.bind(10, 20, b=30, c=40, args=50, kwargs=60)\n    # we won't have 'z' argument in the bound arguments object, as we didn't\n    # pass it to the 'bind'\n    self.assertEqual(tuple(ba.arguments.items()),\n                     (('a', 10), ('args', (20,)), ('b', 30),\n                      ('kwargs', {'c': 40, 'args': 50, 'kwargs': 60})))\n    self.assertEqual(ba.kwargs,\n                     {'b': 30, 'c': 40, 'args': 50, 'kwargs': 60})\n    self.assertEqual(ba.args, (10, 20))\n")
    if sys.version_info[0] > 2:
        exec('\ndef test_signature_bind_positional_only(self):\n    P = inspect.Parameter\n\n    def test(a_po, b_po, c_po=3, foo=42, *, bar=50, **kwargs):\n        return a_po, b_po, c_po, foo, bar, kwargs\n\n    sig = inspect.signature(test)\n    new_params = collections.OrderedDict(tuple(sig.parameters.items()))\n    for name in (\'a_po\', \'b_po\', \'c_po\'):\n        new_params[name] = new_params[name].replace(kind=P.POSITIONAL_ONLY)\n    new_sig = sig.replace(parameters=new_params.values())\n    test.__signature__ = new_sig\n\n    self.assertEqual(self.call(test, 1, 2, 4, 5, bar=6),\n                     (1, 2, 4, 5, 6, {}))\n\n    with self.assertRaisesRegex(TypeError, "parameter is positional only"):\n        self.call(test, 1, 2, c_po=4)\n\n    with self.assertRaisesRegex(TypeError, "parameter is positional only"):\n        self.call(test, a_po=1, b_po=2)\n')

    def test_bind_self(self):

        class F:

            def f(a, self):
                return (a, self)
        an_f = F()
        partial_f = functools.partial(F.f, an_f)
        ba = inspect.signature(partial_f).bind(self=10)
        self.assertEqual((an_f, 10), partial_f(*ba.args, **ba.kwargs))