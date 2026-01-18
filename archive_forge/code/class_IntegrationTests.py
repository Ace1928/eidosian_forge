from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
@skipIf(not isGraphvizModuleInstalled(), 'Graphviz module is not installed.')
@skipIf(not isGraphvizInstalled(), 'Graphviz tools are not installed.')
@skipIf(not isTwistedInstalled(), 'Twisted is not installed.')
class IntegrationTests(TestCase):
    """
    Tests which make sure Graphviz can understand the output produced by
    Automat.
    """

    def test_validGraphviz(self):
        """
        L{graphviz} emits valid graphviz data.
        """
        p = subprocess.Popen('dot', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        out, err = p.communicate(''.join(sampleMachine().asDigraph()).encode('utf-8'))
        self.assertEqual(p.returncode, 0)