import sys
import unittest
import sys
class FrameInfoTest(unittest.TestCase):

    def test_w_module(self):
        from zope.interface.tests import advisory_testing
        kind, module, f_locals, f_globals = advisory_testing.moduleLevelFrameInfo
        self.assertEqual(kind, 'module')
        for d in (module.__dict__, f_locals, f_globals):
            self.assertTrue(d is advisory_testing.my_globals)

    def test_w_class(self):
        from zope.interface.tests import advisory_testing
        kind, module, f_locals, f_globals = advisory_testing.NewStyleClass.classLevelFrameInfo
        self.assertEqual(kind, 'class')
        for d in (module.__dict__, f_globals):
            self.assertTrue(d is advisory_testing.my_globals)

    def test_inside_function_call(self):
        from zope.interface.advice import getFrameInfo
        kind, module, f_locals, f_globals = getFrameInfo(sys._getframe())
        self.assertEqual(kind, 'function call')
        self.assertTrue(f_locals is locals())
        for d in (module.__dict__, f_globals):
            self.assertTrue(d is globals())

    def test_inside_exec(self):
        from zope.interface.advice import getFrameInfo
        _globals = {'getFrameInfo': getFrameInfo}
        _locals = {}
        exec(_FUNKY_EXEC, _globals, _locals)
        self.assertEqual(_locals['kind'], 'exec')
        self.assertTrue(_locals['f_locals'] is _locals)
        self.assertTrue(_locals['module'] is None)
        self.assertTrue(_locals['f_globals'] is _globals)