import unittest as pyunit
from twisted.python.reflect import namedAny
from twisted.trial import unittest
from twisted.trial.test import suppression
def _assertWarnings(self, warnings, which):
    """
        Assert that a certain number of warnings with certain messages were
        emitted in a certain order.

        @param warnings: A list of emitted warnings, as returned by
            C{flushWarnings}.

        @param which: A list of strings giving warning messages that should
            appear in C{warnings}.

        @raise self.failureException: If the warning messages given by C{which}
            do not match the messages in the warning information in C{warnings},
            or if they do not appear in the same order.
        """
    self.assertEqual([warning['message'] for warning in warnings], which)