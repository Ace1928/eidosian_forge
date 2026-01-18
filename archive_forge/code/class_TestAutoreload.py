import os
import platform
import pytest
import sys
import tempfile
import textwrap
import shutil
import random
import time
import traceback
from io import StringIO
from dataclasses import dataclass
import IPython.testing.tools as tt
from unittest import TestCase
from IPython.extensions.autoreload import AutoreloadMagics
from IPython.core.events import EventManager, pre_run_cell
from IPython.testing.decorators import skipif_not_numpy
from IPython.core.interactiveshell import ExecutionInfo
class TestAutoreload(Fixture):

    def test_reload_enums(self):
        mod_name, mod_fn = self.new_module(textwrap.dedent("\n                                from enum import Enum\n                                class MyEnum(Enum):\n                                    A = 'A'\n                                    B = 'B'\n                            "))
        self.shell.magic_autoreload('2')
        self.shell.magic_aimport(mod_name)
        self.write_file(mod_fn, textwrap.dedent("\n                                from enum import Enum\n                                class MyEnum(Enum):\n                                    A = 'A'\n                                    B = 'B'\n                                    C = 'C'\n                            "))
        with tt.AssertNotPrints('[autoreload of %s failed:' % mod_name, channel='stderr'):
            self.shell.run_code('pass')

    def test_reload_class_type(self):
        self.shell.magic_autoreload('2')
        mod_name, mod_fn = self.new_module('\n            class Test():\n                def meth(self):\n                    return "old"\n        ')
        assert 'test' not in self.shell.ns
        assert 'result' not in self.shell.ns
        self.shell.run_code('from %s import Test' % mod_name)
        self.shell.run_code('test = Test()')
        self.write_file(mod_fn, '\n            class Test():\n                def meth(self):\n                    return "new"\n        ')
        test_object = self.shell.ns['test']
        self.shell.run_code('pass')
        test_class = pickle_get_current_class(test_object)
        assert isinstance(test_object, test_class)
        self.shell.run_code('import pickle')
        self.shell.run_code('p = pickle.dumps(test)')

    def test_reload_class_attributes(self):
        self.shell.magic_autoreload('2')
        mod_name, mod_fn = self.new_module(textwrap.dedent("\n                                class MyClass:\n\n                                    def __init__(self, a=10):\n                                        self.a = a\n                                        self.b = 22 \n                                        # self.toto = 33\n\n                                    def square(self):\n                                        print('compute square')\n                                        return self.a*self.a\n                            "))
        self.shell.run_code('from %s import MyClass' % mod_name)
        self.shell.run_code('first = MyClass(5)')
        self.shell.run_code('first.square()')
        with self.assertRaises(AttributeError):
            self.shell.run_code('first.cube()')
        with self.assertRaises(AttributeError):
            self.shell.run_code('first.power(5)')
        self.shell.run_code('first.b')
        with self.assertRaises(AttributeError):
            self.shell.run_code('first.toto')
        self.write_file(mod_fn, textwrap.dedent("\n                            class MyClass:\n\n                                def __init__(self, a=10):\n                                    self.a = a\n                                    self.b = 11\n\n                                def power(self, p):\n                                    print('compute power '+str(p))\n                                    return self.a**p\n                            "))
        self.shell.run_code('second = MyClass(5)')
        for object_name in {'first', 'second'}:
            self.shell.run_code(f'{object_name}.power(5)')
            with self.assertRaises(AttributeError):
                self.shell.run_code(f'{object_name}.cube()')
            with self.assertRaises(AttributeError):
                self.shell.run_code(f'{object_name}.square()')
            self.shell.run_code(f'{object_name}.b')
            self.shell.run_code(f'{object_name}.a')
            with self.assertRaises(AttributeError):
                self.shell.run_code(f'{object_name}.toto')

    @skipif_not_numpy
    def test_comparing_numpy_structures(self):
        self.shell.magic_autoreload('2')
        self.shell.run_code('1+1')
        mod_name, mod_fn = self.new_module(textwrap.dedent('\n                                import numpy as np\n                                class MyClass:\n                                    a = (np.array((.1, .2)),\n                                         np.array((.2, .3)))\n                            '))
        self.shell.run_code('from %s import MyClass' % mod_name)
        self.shell.run_code('first = MyClass()')
        self.write_file(mod_fn, textwrap.dedent('\n                                import numpy as np\n                                class MyClass:\n                                    a = (np.array((.3, .4)),\n                                         np.array((.5, .6)))\n                            '))
        with tt.AssertNotPrints('[autoreload of %s failed:' % mod_name, channel='stderr'):
            self.shell.run_code('pass')

    def test_autoload_newly_added_objects(self):
        self.shell.magic_autoreload('3')
        mod_code = '\n        def func1(): pass\n        '
        mod_name, mod_fn = self.new_module(textwrap.dedent(mod_code))
        self.shell.run_code(f'from {mod_name} import *')
        self.shell.run_code('func1()')
        with self.assertRaises(NameError):
            self.shell.run_code('func2()')
        with self.assertRaises(NameError):
            self.shell.run_code('t = Test()')
        with self.assertRaises(NameError):
            self.shell.run_code('number')
        new_code = "\n        def func1(): pass\n        def func2(): pass\n        class Test: pass\n        number = 0\n        from enum import Enum\n        class TestEnum(Enum):\n            A = 'a'\n        "
        self.write_file(mod_fn, textwrap.dedent(new_code))
        self.shell.run_code('func2()')
        self.shell.run_code(f"import sys; sys.modules['{mod_name}'].func2()")
        self.shell.run_code('t = Test()')
        self.shell.run_code('number')
        self.shell.run_code('TestEnum.A')
        new_code = "\n        def func1(): return 'changed'\n        def func2(): return 'changed'\n        class Test:\n            def new_func(self):\n                return 'changed'\n        number = 1\n        from enum import Enum\n        class TestEnum(Enum):\n            A = 'a'\n            B = 'added'\n        "
        self.write_file(mod_fn, textwrap.dedent(new_code))
        self.shell.run_code("assert func1() == 'changed'")
        self.shell.run_code("assert func2() == 'changed'")
        self.shell.run_code("t = Test(); assert t.new_func() == 'changed'")
        self.shell.run_code('assert number == 1')
        if sys.version_info < (3, 12):
            self.shell.run_code("assert TestEnum.B.value == 'added'")
        new_mod_code = "\n        from enum import Enum\n        class Ext(Enum):\n            A = 'ext'\n        def ext_func():\n            return 'ext'\n        class ExtTest:\n            def meth(self):\n                return 'ext'\n        ext_int = 2\n        "
        new_mod_name, new_mod_fn = self.new_module(textwrap.dedent(new_mod_code))
        current_mod_code = f'\n        from {new_mod_name} import *\n        '
        self.write_file(mod_fn, textwrap.dedent(current_mod_code))
        self.shell.run_code("assert Ext.A.value == 'ext'")
        self.shell.run_code("assert ext_func() == 'ext'")
        self.shell.run_code("t = ExtTest(); assert t.meth() == 'ext'")
        self.shell.run_code('assert ext_int == 2')

    def test_verbose_names(self):

        @dataclass
        class AutoreloadSettings:
            check_all: bool
            enabled: bool
            autoload_obj: bool

        def gather_settings(mode):
            self.shell.magic_autoreload(mode)
            module_reloader = self.shell.auto_magics._reloader
            return AutoreloadSettings(module_reloader.check_all, module_reloader.enabled, module_reloader.autoload_obj)
        assert gather_settings('0') == gather_settings('off')
        assert gather_settings('0') == gather_settings('OFF')
        assert gather_settings('1') == gather_settings('explicit')
        assert gather_settings('2') == gather_settings('all')
        assert gather_settings('3') == gather_settings('complete')
        with self.assertRaises(ValueError):
            self.shell.magic_autoreload('4')

    def test_aimport_parsing(self):
        module_reloader = self.shell.auto_magics._reloader
        self.shell.magic_aimport('os')
        assert module_reloader.modules['os'] is True
        assert 'os' not in module_reloader.skip_modules.keys()
        self.shell.magic_aimport('-math')
        assert module_reloader.skip_modules['math'] is True
        assert 'math' not in module_reloader.modules.keys()
        self.shell.magic_aimport('-os, math')
        assert module_reloader.modules['math'] is True
        assert 'math' not in module_reloader.skip_modules.keys()
        assert module_reloader.skip_modules['os'] is True
        assert 'os' not in module_reloader.modules.keys()

    def test_autoreload_output(self):
        self.shell.magic_autoreload('complete')
        mod_code = '\n        def func1(): pass\n        '
        mod_name, mod_fn = self.new_module(mod_code)
        self.shell.run_code(f'import {mod_name}')
        with tt.AssertPrints('', channel='stdout'):
            self.shell.run_code('pass')
        self.shell.magic_autoreload('complete --print')
        self.write_file(mod_fn, mod_code)
        with tt.AssertPrints(f"Reloading '{mod_name}'.", channel='stdout'):
            self.shell.run_code('pass')
        self.shell.magic_autoreload('complete -p')
        self.write_file(mod_fn, mod_code)
        with tt.AssertPrints(f"Reloading '{mod_name}'.", channel='stdout'):
            self.shell.run_code('pass')
        self.shell.magic_autoreload('complete --print --log')
        self.write_file(mod_fn, mod_code)
        with tt.AssertPrints(f"Reloading '{mod_name}'.", channel='stdout'):
            self.shell.run_code('pass')
        self.shell.magic_autoreload('complete --print --log')
        self.write_file(mod_fn, mod_code)
        with self.assertLogs(logger='autoreload') as lo:
            self.shell.run_code('pass')
        assert lo.output == [f"INFO:autoreload:Reloading '{mod_name}'."]
        self.shell.magic_autoreload('complete -l')
        self.write_file(mod_fn, mod_code)
        with self.assertLogs(logger='autoreload') as lo:
            self.shell.run_code('pass')
        assert lo.output == [f"INFO:autoreload:Reloading '{mod_name}'."]

    def _check_smoketest(self, use_aimport=True):
        """
        Functional test for the automatic reloader using either
        '%autoreload 1' or '%autoreload 2'
        """
        mod_name, mod_fn = self.new_module("\nx = 9\n\nz = 123  # this item will be deleted\n\ndef foo(y):\n    return y + 3\n\nclass Baz(object):\n    def __init__(self, x):\n        self.x = x\n    def bar(self, y):\n        return self.x + y\n    @property\n    def quux(self):\n        return 42\n    def zzz(self):\n        '''This method will be deleted below'''\n        return 99\n\nclass Bar:    # old-style class: weakref doesn't work for it on Python < 2.7\n    def foo(self):\n        return 1\n")
        if use_aimport:
            self.shell.magic_autoreload('1')
            self.shell.magic_aimport(mod_name)
            stream = StringIO()
            self.shell.magic_aimport('', stream=stream)
            self.assertIn('Modules to reload:\n%s' % mod_name, stream.getvalue())
            with self.assertRaises(ImportError):
                self.shell.magic_aimport('tmpmod_as318989e89ds')
        else:
            self.shell.magic_autoreload('2')
            self.shell.run_code('import %s' % mod_name)
            stream = StringIO()
            self.shell.magic_aimport('', stream=stream)
            self.assertTrue('Modules to reload:\nall-except-skipped' in stream.getvalue())
        self.assertIn(mod_name, self.shell.ns)
        mod = sys.modules[mod_name]
        old_foo = mod.foo
        old_obj = mod.Baz(9)
        old_obj2 = mod.Bar()

        def check_module_contents():
            self.assertEqual(mod.x, 9)
            self.assertEqual(mod.z, 123)
            self.assertEqual(old_foo(0), 3)
            self.assertEqual(mod.foo(0), 3)
            obj = mod.Baz(9)
            self.assertEqual(old_obj.bar(1), 10)
            self.assertEqual(obj.bar(1), 10)
            self.assertEqual(obj.quux, 42)
            self.assertEqual(obj.zzz(), 99)
            obj2 = mod.Bar()
            self.assertEqual(old_obj2.foo(), 1)
            self.assertEqual(obj2.foo(), 1)
        check_module_contents()
        self.write_file(mod_fn, '\na syntax error\n')
        with tt.AssertPrints('[autoreload of %s failed:' % mod_name, channel='stderr'):
            self.shell.run_code('pass')
        with tt.AssertNotPrints('[autoreload of %s failed:' % mod_name, channel='stderr'):
            self.shell.run_code('pass')
        check_module_contents()
        self.write_file(mod_fn, '\nx = 10\n\ndef foo(y):\n    return y + 4\n\nclass Baz(object):\n    def __init__(self, x):\n        self.x = x\n    def bar(self, y):\n        return self.x + y + 1\n    @property\n    def quux(self):\n        return 43\n\nclass Bar:    # old-style class\n    def foo(self):\n        return 2\n')

        def check_module_contents():
            self.assertEqual(mod.x, 10)
            self.assertFalse(hasattr(mod, 'z'))
            self.assertEqual(old_foo(0), 4)
            self.assertEqual(mod.foo(0), 4)
            obj = mod.Baz(9)
            self.assertEqual(old_obj.bar(1), 11)
            self.assertEqual(obj.bar(1), 11)
            self.assertEqual(old_obj.quux, 43)
            self.assertEqual(obj.quux, 43)
            self.assertFalse(hasattr(old_obj, 'zzz'))
            self.assertFalse(hasattr(obj, 'zzz'))
            obj2 = mod.Bar()
            self.assertEqual(old_obj2.foo(), 2)
            self.assertEqual(obj2.foo(), 2)
        self.shell.run_code('pass')
        check_module_contents()
        os.unlink(mod_fn)
        self.shell.run_code('pass')
        check_module_contents()
        if use_aimport:
            self.shell.magic_aimport('-' + mod_name)
            stream = StringIO()
            self.shell.magic_aimport('', stream=stream)
            self.assertTrue('Modules to skip:\n%s' % mod_name in stream.getvalue())
            self.shell.magic_aimport('-tmpmod_as318989e89ds')
        else:
            self.shell.magic_autoreload('0')
        self.write_file(mod_fn, '\nx = -99\n')
        self.shell.run_code('pass')
        self.shell.run_code('pass')
        check_module_contents()
        if use_aimport:
            self.shell.magic_aimport(mod_name)
        else:
            self.shell.magic_autoreload('')
        self.shell.run_code('pass')
        self.assertEqual(mod.x, -99)

    def test_smoketest_aimport(self):
        self._check_smoketest(use_aimport=True)

    def test_smoketest_autoreload(self):
        self._check_smoketest(use_aimport=False)