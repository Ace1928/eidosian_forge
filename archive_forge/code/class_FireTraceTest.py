from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
class FireTraceTest(testutils.BaseTestCase):

    def testFireTraceInitialization(self):
        t = trace.FireTrace(10)
        self.assertIsNotNone(t)
        self.assertIsNotNone(t.elements)

    def testFireTraceGetResult(self):
        t = trace.FireTrace('start')
        self.assertEqual(t.GetResult(), 'start')
        t.AddAccessedProperty('t', 'final', None, 'example.py', 10)
        self.assertEqual(t.GetResult(), 't')

    def testFireTraceHasError(self):
        t = trace.FireTrace('start')
        self.assertFalse(t.HasError())
        t.AddAccessedProperty('t', 'final', None, 'example.py', 10)
        self.assertFalse(t.HasError())
        t.AddError(ValueError('example error'), ['arg'])
        self.assertTrue(t.HasError())

    def testAddAccessedProperty(self):
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddAccessedProperty('new component', 'prop', args, 'sample.py', 12)
        self.assertEqual(str(t), '1. Initial component\n2. Accessed property "prop" (sample.py:12)')

    def testAddCalledCallable(self):
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('result', 'cell', args, 'sample.py', 10, False, action=trace.CALLED_CALLABLE)
        self.assertEqual(str(t), '1. Initial component\n2. Called callable "cell" (sample.py:10)')

    def testAddCalledRoutine(self):
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(str(t), '1. Initial component\n2. Called routine "run" (sample.py:12)')

    def testAddInstantiatedClass(self):
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('Classname', 'classname', args, 'sample.py', 12, False, action=trace.INSTANTIATED_CLASS)
        target = '1. Initial component\n2. Instantiated class "classname" (sample.py:12)'
        self.assertEqual(str(t), target)

    def testAddCompletionScript(self):
        t = trace.FireTrace('initial object')
        t.AddCompletionScript('This is the completion script string.')
        self.assertEqual(str(t), '1. Initial component\n2. Generated completion script')

    def testAddInteractiveMode(self):
        t = trace.FireTrace('initial object')
        t.AddInteractiveMode()
        self.assertEqual(str(t), '1. Initial component\n2. Entered interactive mode')

    def testGetCommand(self):
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(t.GetCommand(), 'example args')

    def testGetCommandWithQuotes(self):
        t = trace.FireTrace('initial object')
        args = ('example', 'spaced arg')
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(t.GetCommand(), "example 'spaced arg'")

    def testGetCommandWithFlagQuotes(self):
        t = trace.FireTrace('initial object')
        args = ('--example=spaced arg',)
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(t.GetCommand(), "--example='spaced arg'")