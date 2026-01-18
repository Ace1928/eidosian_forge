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
class TableMakerTests(TestCase):
    """
    Tests that ensure L{tableMaker} generates HTML tables usable as
    labels in DOT graphs.

    For more information, read the "HTML-Like Labels" section of
    U{http://www.graphviz.org/doc/info/shapes.html}.
    """

    def fakeElementMaker(self, name, *children, **attributes):
        return HTMLElement(name=name, children=children, attributes=attributes)

    def setUp(self):
        from .._visualize import tableMaker
        self.inputLabel = 'input label'
        self.port = 'the port'
        self.tableMaker = functools.partial(tableMaker, _E=self.fakeElementMaker)

    def test_inputLabelRow(self):
        """
        The table returned by L{tableMaker} always contains the input
        symbol label in its first row, and that row contains one cell
        with a port attribute set to the provided port.
        """

        def hasPort(element):
            return not isLeaf(element) and element.attributes.get('port') == self.port
        for outputLabels in ([], ['an output label']):
            table = self.tableMaker(self.inputLabel, outputLabels, port=self.port)
            self.assertGreater(len(table.children), 0)
            inputLabelRow = table.children[0]
            portCandidates = findElements(table, hasPort)
            self.assertEqual(len(portCandidates), 1)
            self.assertEqual(portCandidates[0].name, 'td')
            self.assertEqual(findElements(inputLabelRow, isLeaf), [self.inputLabel])

    def test_noOutputLabels(self):
        """
        L{tableMaker} does not add a colspan attribute to the input
        label's cell or a second row if there no output labels.
        """
        table = self.tableMaker('input label', (), port=self.port)
        self.assertEqual(len(table.children), 1)
        inputLabelRow, = table.children
        self.assertNotIn('colspan', inputLabelRow.attributes)

    def test_withOutputLabels(self):
        """
        L{tableMaker} adds a colspan attribute to the input label's cell
        equal to the number of output labels and a second row that
        contains the output labels.
        """
        table = self.tableMaker(self.inputLabel, ('output label 1', 'output label 2'), port=self.port)
        self.assertEqual(len(table.children), 2)
        inputRow, outputRow = table.children

        def hasCorrectColspan(element):
            return not isLeaf(element) and element.name == 'td' and (element.attributes.get('colspan') == '2')
        self.assertEqual(len(findElements(inputRow, hasCorrectColspan)), 1)
        self.assertEqual(findElements(outputRow, isLeaf), ['output label 1', 'output label 2'])