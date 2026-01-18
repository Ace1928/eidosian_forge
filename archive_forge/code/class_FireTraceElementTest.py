from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
class FireTraceElementTest(testutils.BaseTestCase):

    def testFireTraceElementHasError(self):
        el = trace.FireTraceElement()
        self.assertFalse(el.HasError())
        el = trace.FireTraceElement(error=ValueError('example error'))
        self.assertTrue(el.HasError())

    def testFireTraceElementAsStringNoMetadata(self):
        el = trace.FireTraceElement(component='Example', action='Fake action')
        self.assertEqual(str(el), 'Fake action')

    def testFireTraceElementAsStringWithTarget(self):
        el = trace.FireTraceElement(component='Example', action='Created toy', target='Beaker')
        self.assertEqual(str(el), 'Created toy "Beaker"')

    def testFireTraceElementAsStringWithTargetAndLineNo(self):
        el = trace.FireTraceElement(component='Example', action='Created toy', target='Beaker', filename='beaker.py', lineno=10)
        self.assertEqual(str(el), 'Created toy "Beaker" (beaker.py:10)')