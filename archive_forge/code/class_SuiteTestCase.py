import doctest
import os
import pickle
import sys
from tempfile import mkstemp
import unittest
from genshi.core import Markup
from genshi.template.base import Context
from genshi.template.eval import Expression, Suite, Undefined, UndefinedError, \
from genshi.compat import BytesIO, IS_PYTHON2, wrapped_bytes
class SuiteTestCase(unittest.TestCase):

    def test_pickle(self):
        suite = Suite('foo = 42')
        buf = BytesIO()
        pickle.dump(suite, buf, 2)
        buf.seek(0)
        unpickled = pickle.load(buf)
        data = {}
        unpickled.execute(data)
        self.assertEqual(42, data['foo'])
        assert unpickled.code == suite.code

    def test_internal_shadowing(self):
        suite = Suite('data = []\nbar = foo\n')
        data = {'foo': 42}
        suite.execute(data)
        self.assertEqual(42, data['bar'])

    def test_assign(self):
        suite = Suite('foo = 42')
        data = {}
        suite.execute(data)
        self.assertEqual(42, data['foo'])

    def test_def(self):
        suite = Suite('def donothing(): pass')
        data = {}
        suite.execute(data)
        assert 'donothing' in data
        self.assertEqual(None, data['donothing']())

    def test_def_with_multiple_statements(self):
        suite = Suite('\ndef donothing():\n    if True:\n        return foo\n')
        data = {'foo': 'bar'}
        suite.execute(data)
        assert 'donothing' in data
        self.assertEqual('bar', data['donothing']())

    def test_def_using_nonlocal(self):
        suite = Suite("\nvalues = []\ndef add(value):\n    if value not in values:\n        values.append(value)\nadd('foo')\nadd('bar')\n")
        data = {}
        suite.execute(data)
        self.assertEqual(['foo', 'bar'], data['values'])

    def test_def_some_defaults(self):
        suite = Suite('\ndef difference(v1, v2=10):\n    return v1 - v2\nx = difference(20, 19)\ny = difference(20)\n')
        data = {}
        suite.execute(data)
        self.assertEqual(1, data['x'])
        self.assertEqual(10, data['y'])

    def test_def_all_defaults(self):
        suite = Suite('\ndef difference(v1=100, v2=10):\n    return v1 - v2\nx = difference(20, 19)\ny = difference(20)\nz = difference()\n')
        data = {}
        suite.execute(data)
        self.assertEqual(1, data['x'])
        self.assertEqual(10, data['y'])
        self.assertEqual(90, data['z'])

    def test_def_vararg(self):
        suite = Suite('\ndef mysum(*others):\n    rv = 0\n    for n in others:\n        rv = rv + n\n    return rv\nx = mysum(1, 2, 3)\n')
        data = {}
        suite.execute(data)
        self.assertEqual(6, data['x'])

    def test_def_kwargs(self):
        suite = Suite("\ndef smash(**kw):\n    return [''.join(i) for i in kw.items()]\nx = smash(foo='abc', bar='def')\n")
        data = {}
        suite.execute(data)
        self.assertEqual(['bardef', 'fooabc'], sorted(data['x']))
    if not IS_PYTHON2:

        def test_def_kwonlyarg(self):
            suite = Suite('\ndef kwonly(*args, k):\n    return k\nx = kwonly(k="foo")\n')
            data = {}
            suite.execute(data)
            self.assertEqual('foo', data['x'])

        def test_def_kwonlyarg_with_default(self):
            suite = Suite('\ndef kwonly(*args, k="bar"):\n    return k\nx = kwonly(k="foo")\ny = kwonly()\n')
            data = {}
            suite.execute(data)
            self.assertEqual('foo', data['x'])
            self.assertEqual('bar', data['y'])

    def test_def_nested(self):
        suite = Suite("\ndef doit():\n    values = []\n    def add(value):\n        if value not in values:\n            values.append(value)\n    add('foo')\n    add('bar')\n    return values\nx = doit()\n")
        data = {}
        suite.execute(data)
        self.assertEqual(['foo', 'bar'], data['x'])

    def test_def_with_decorator(self):
        suite = Suite("\ndef lower(fun):\n    return lambda: fun().lower()\n\n@lower\ndef say_hi():\n    return 'Hi!'\n\nresult = say_hi()\n")
        data = {}
        suite.execute(data)
        self.assertEqual('hi!', data['result'])

    def test_delete(self):
        suite = Suite('foo = 42\ndel foo\n')
        data = {}
        suite.execute(data)
        assert 'foo' not in data

    def test_class(self):
        suite = Suite('class plain(object): pass')
        data = {}
        suite.execute(data)
        assert 'plain' in data

    def test_class_in_def(self):
        suite = Suite("\ndef create():\n    class Foobar(object):\n        def __str__(self):\n            return 'foobar'\n    return Foobar()\nx = create()\n")
        data = {}
        suite.execute(data)
        self.assertEqual('foobar', str(data['x']))

    def test_class_with_methods(self):
        suite = Suite('class plain(object):\n    def donothing():\n        pass\n')
        data = {}
        suite.execute(data)
        assert 'plain' in data

    def test_import(self):
        suite = Suite('from itertools import repeat')
        data = {}
        suite.execute(data)
        assert 'repeat' in data

    def test_import_star(self):
        suite = Suite('from itertools import *')
        data = Context()
        suite.execute(data)
        assert 'repeat' in data

    def test_import_in_def(self):
        suite = Suite('def fun():\n    from itertools import repeat\n    return repeat(1, 3)\n')
        data = Context()
        suite.execute(data)
        assert 'repeat' not in data
        self.assertEqual([1, 1, 1], list(data['fun']()))

    def test_for(self):
        suite = Suite('x = []\nfor i in range(3):\n    x.append(i**2)\n')
        data = {}
        suite.execute(data)
        self.assertEqual([0, 1, 4], data['x'])

    def test_for_in_def(self):
        suite = Suite('def loop():\n    for i in range(10):\n        if i == 5:\n            break\n    return i\n')
        data = {}
        suite.execute(data)
        assert 'loop' in data
        self.assertEqual(5, data['loop']())

    def test_if(self):
        suite = Suite('if foo == 42:\n    x = True\n')
        data = {'foo': 42}
        suite.execute(data)
        self.assertEqual(True, data['x'])

    def test_raise(self):
        suite = Suite('raise NotImplementedError')
        self.assertRaises(NotImplementedError, suite.execute, {})

    def test_try_except(self):
        suite = Suite('try:\n    import somemod\nexcept ImportError:\n    somemod = None\nelse:\n    somemod.dosth()')
        data = {}
        suite.execute(data)
        self.assertEqual(None, data['somemod'])

    def test_finally(self):
        suite = Suite('try:\n    x = 2\nfinally:\n    x = None\n')
        data = {}
        suite.execute(data)
        self.assertEqual(None, data['x'])

    def test_while_break(self):
        suite = Suite('x = 0\nwhile x < 5:\n    x += step\n    if x == 4:\n        break\n')
        data = {'step': 2}
        suite.execute(data)
        self.assertEqual(4, data['x'])

    def test_augmented_attribute_assignment(self):
        suite = Suite("d['k'] += 42")
        d = {'k': 1}
        suite.execute({'d': d})
        self.assertEqual(43, d['k'])

    def test_local_augmented_assign(self):
        Suite('x = 1; x += 42; assert x == 43').execute({})

    def test_augmented_assign_in_def(self):
        d = {}
        Suite('def foo():\n    i = 1\n    i += 1\n    return i\nx = foo()').execute(d)
        self.assertEqual(2, d['x'])

    def test_augmented_assign_in_loop_in_def(self):
        d = {}
        Suite('def foo():\n    i = 0\n    for n in range(5):\n        i += n\n    return i\nx = foo()').execute(d)
        self.assertEqual(10, d['x'])

    def test_assign_in_list(self):
        suite = Suite("[d['k']] = 'foo',; assert d['k'] == 'foo'")
        d = {'k': 'bar'}
        suite.execute({'d': d})
        self.assertEqual('foo', d['k'])

    def test_exec(self):
        suite = Suite("x = 1; exec(d['k']); assert x == 42, x")
        suite.execute({'d': {'k': 'x = 42'}})

    def test_return(self):
        suite = Suite('\ndef f():\n    return v\n\nassert f() == 42\n')
        suite.execute({'v': 42})

    def test_assign_to_dict_item(self):
        suite = Suite("d['k'] = 'foo'")
        data = {'d': {}}
        suite.execute(data)
        self.assertEqual('foo', data['d']['k'])

    def test_assign_to_attribute(self):

        class Something(object):
            pass
        something = Something()
        suite = Suite("obj.attr = 'foo'")
        data = {'obj': something}
        suite.execute(data)
        self.assertEqual('foo', something.attr)

    def test_delattr(self):

        class Something(object):

            def __init__(self):
                self.attr = 'foo'
        obj = Something()
        Suite('del obj.attr').execute({'obj': obj})
        self.assertFalse(hasattr(obj, 'attr'))

    def test_delitem(self):
        d = {'k': 'foo'}
        Suite("del d['k']").execute({'d': d})
        self.assertNotIn('k', d)

    def test_with_statement(self):
        fd, path = mkstemp()
        f = os.fdopen(fd, 'w')
        try:
            f.write('foo\nbar\n')
            f.seek(0)
            f.close()
            d = {'path': path}
            suite = Suite('from __future__ import with_statement\nlines = []\nwith open(path) as file:\n    for line in file:\n        lines.append(line)\n')
            suite.execute(d)
            self.assertEqual(['foo\n', 'bar\n'], d['lines'])
        finally:
            os.remove(path)

    def test_yield_expression(self):
        d = {}
        suite = Suite('\nresults = []\ndef counter(maximum):\n    i = 0\n    while i < maximum:\n        val = (yield i)\n        if val is not None:\n            i = val\n        else:\n            i += 1\nit = counter(5)\nresults.append(next(it))\nresults.append(it.send(3))\nresults.append(next(it))\n')
        suite.execute(d)
        self.assertEqual([0, 3, 4], d['results'])
    if sys.version_info >= (3, 3):

        def test_with_statement_with_multiple_items(self):
            fd, path = mkstemp()
            f = os.fdopen(fd, 'w')
            try:
                f.write('foo\n')
                f.seek(0)
                f.close()
                d = {'path': path}
                suite = Suite('from __future__ import with_statement\nlines = []\nwith open(path) as file1, open(path) as file2:\n    for line in file1:\n        lines.append(line)\n    for line in file2:\n        lines.append(line)\n')
                suite.execute(d)
                self.assertEqual(['foo\n', 'foo\n'], d['lines'])
            finally:
                os.remove(path)

    def test_slice(self):
        suite = Suite('x = numbers[0:2]')
        data = {'numbers': [0, 1, 2, 3]}
        suite.execute(data)
        self.assertEqual([0, 1], data['x'])

    def test_slice_with_vars(self):
        suite = Suite('x = numbers[start:end]')
        data = {'numbers': [0, 1, 2, 3], 'start': 0, 'end': 2}
        suite.execute(data)
        self.assertEqual([0, 1], data['x'])

    def test_slice_copy(self):
        suite = Suite('x = numbers[:]')
        data = {'numbers': [0, 1, 2, 3]}
        suite.execute(data)
        self.assertEqual([0, 1, 2, 3], data['x'])

    def test_slice_stride(self):
        suite = Suite('x = numbers[::stride]')
        data = {'numbers': [0, 1, 2, 3, 4], 'stride': 2}
        suite.execute(data)
        self.assertEqual([0, 2, 4], data['x'])

    def test_slice_negative_start(self):
        suite = Suite('x = numbers[-1:]')
        data = {'numbers': [0, 1, 2, 3, 4], 'stride': 2}
        suite.execute(data)
        self.assertEqual([4], data['x'])

    def test_slice_negative_end(self):
        suite = Suite('x = numbers[:-1]')
        data = {'numbers': [0, 1, 2, 3, 4], 'stride': 2}
        suite.execute(data)
        self.assertEqual([0, 1, 2, 3], data['x'])

    def test_slice_constant(self):
        suite = Suite('x = numbers[1]')
        data = {'numbers': [0, 1, 2, 3, 4]}
        suite.execute(data)
        self.assertEqual(1, data['x'])

    def test_slice_call(self):

        def f():
            return 2
        suite = Suite('x = numbers[f()]')
        data = {'numbers': [0, 1, 2, 3, 4], 'f': f}
        suite.execute(data)
        self.assertEqual(2, data['x'])

    def test_slice_name(self):
        suite = Suite('x = numbers[v]')
        data = {'numbers': [0, 1, 2, 3, 4], 'v': 2}
        suite.execute(data)
        self.assertEqual(2, data['x'])

    def test_slice_attribute(self):

        class ValueHolder:

            def __init__(self):
                self.value = 3
        suite = Suite('x = numbers[obj.value]')
        data = {'numbers': [0, 1, 2, 3, 4], 'obj': ValueHolder()}
        suite.execute(data)
        self.assertEqual(3, data['x'])