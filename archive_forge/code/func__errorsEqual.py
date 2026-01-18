import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def _errorsEqual(self, has_invariant, error_len, error_msgs, iface):
    from zope.interface.exceptions import Invalid
    self.assertRaises(Invalid, iface.validateInvariants, has_invariant)
    e = []
    try:
        iface.validateInvariants(has_invariant, e)
        self.fail('validateInvariants should always raise')
    except Invalid as error:
        self.assertEqual(error.args[0], e)
    self.assertEqual(len(e), error_len)
    msgs = [error.args[0] for error in e]
    msgs.sort()
    for msg in msgs:
        self.assertEqual(msg, error_msgs.pop(0))