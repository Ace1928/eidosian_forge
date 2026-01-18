from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
@skipIf(not isGraphvizModuleInstalled(), 'Graphviz module is not installed.')
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class SpotChecks(TestCase):
    """
    Tests to make sure that the output contains salient features of the machine
    being generated.
    """

    def test_containsMachineFeatures(self):
        """
        The output of L{graphviz} should contain the names of the states,
        inputs, outputs in the state machine.
        """
        gvout = ''.join(sampleMachine().asDigraph())
        self.assertIn('begin', gvout)
        self.assertIn('end', gvout)
        self.assertIn('go', gvout)
        self.assertIn('out', gvout)