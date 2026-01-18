import unittest
import logging
import os
from prov.model import ProvDocument, ProvBundle, ProvException, first, Literal
from prov.tests import examples
from prov.tests.attributes import TestAttributesBase
from prov.tests.qnames import TestQualifiedNamesBase
from prov.tests.statements import TestStatementsBase
from prov.tests.utility import RoundTripTestCase
class TestAddBundle(unittest.TestCase):

    def document_1(self):
        d1 = ProvDocument()
        ns_ex = d1.add_namespace('ex', EX_URI)
        d1.entity(ns_ex['e1'])
        return d1

    def document_2(self):
        d2 = ProvDocument()
        ns_ex = d2.add_namespace('ex', EX2_URI)
        d2.activity(ns_ex['a1'])
        return d2

    def bundle_0(self):
        b = ProvBundle(namespaces={'ex': EX2_URI})
        return b

    def test_add_bundle_simple(self):
        d1 = self.document_1()
        b0 = self.bundle_0()

        def sub_test_1():
            d1.add_bundle(b0)
        self.assertRaises(ProvException, sub_test_1)
        self.assertFalse(d1.has_bundles())
        d1.add_bundle(b0, 'ex:b0')
        self.assertTrue(d1.has_bundles())
        self.assertIn(b0, d1.bundles)

        def sub_test_2():
            ex2_b0 = b0.identifier
            d1.add_bundle(ProvBundle(identifier=ex2_b0))
        self.assertRaises(ProvException, sub_test_2)
        d1.add_bundle(ProvBundle(), 'ex:b0')
        self.assertEqual(len(d1.bundles), 2)

    def test_add_bundle_document(self):
        d1 = self.document_1()
        d2 = self.document_2()

        def sub_test_1():
            d1.add_bundle(d2)
        self.assertRaises(ProvException, sub_test_1)
        ex2_b2 = d2.valid_qualified_name('ex:b2')
        d1.add_bundle(d2, 'ex:b2')
        self.assertEqual(ex2_b2, first(d1.bundles).identifier)
        self.assertNotIn(d2, d1.bundles)
        b2 = ProvBundle()
        b2.update(d2)
        self.assertIn(b2, d1.bundles)