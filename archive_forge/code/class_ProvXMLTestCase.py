import difflib
import glob
import inspect
import io
from lxml import etree
import os
import unittest
import warnings
from prov.identifier import Namespace, QualifiedName
from prov.constants import PROV
import prov.model as prov
from prov.tests.test_model import AllTestsBase
from prov.tests.utility import RoundTripTestCase
class ProvXMLTestCase(unittest.TestCase):

    def test_serialization_example_6(self):
        """
        Test the serialization of example 6 which is a simple entity
        description.
        """
        document = prov.ProvDocument()
        ex_ns = document.add_namespace(*EX_NS)
        document.add_namespace(*EX_TR)
        document.entity('tr:WD-prov-dm-20111215', ((prov.PROV_TYPE, ex_ns['Document']), ('ex:version', '2')))
        with io.BytesIO() as actual:
            document.serialize(format='xml', destination=actual)
            compare_xml(os.path.join(DATA_PATH, 'example_06.xml'), actual)

    def test_serialization_example_7(self):
        """
        Test the serialization of example 7 which is a basic activity.
        """
        document = prov.ProvDocument()
        document.add_namespace(*EX_NS)
        document.activity('ex:a1', '2011-11-16T16:05:00', '2011-11-16T16:06:00', [(prov.PROV_TYPE, prov.Literal('ex:edit', prov.XSD_QNAME)), ('ex:host', 'server.example.org')])
        with io.BytesIO() as actual:
            document.serialize(format='xml', destination=actual)
            compare_xml(os.path.join(DATA_PATH, 'example_07.xml'), actual)

    def test_serialization_example_8(self):
        """
        Test the serialization of example 8 which deals with generation.
        """
        document = prov.ProvDocument()
        document.add_namespace(*EX_NS)
        e1 = document.entity('ex:e1')
        a1 = document.activity('ex:a1')
        document.wasGeneratedBy(entity=e1, activity=a1, time='2001-10-26T21:32:52', other_attributes={'ex:port': 'p1'})
        e2 = document.entity('ex:e2')
        document.wasGeneratedBy(entity=e2, activity=a1, time='2001-10-26T10:00:00', other_attributes={'ex:port': 'p2'})
        with io.BytesIO() as actual:
            document.serialize(format='xml', destination=actual)
            compare_xml(os.path.join(DATA_PATH, 'example_08.xml'), actual)

    def test_deserialization_example_6(self):
        """
        Test the deserialization of example 6 which is a simple entity
        description.
        """
        actual_doc = prov.ProvDocument.deserialize(source=os.path.join(DATA_PATH, 'example_06.xml'), format='xml')
        expected_document = prov.ProvDocument()
        ex_ns = expected_document.add_namespace(*EX_NS)
        expected_document.add_namespace(*EX_TR)
        expected_document.entity('tr:WD-prov-dm-20111215', ((prov.PROV_TYPE, ex_ns['Document']), ('ex:version', '2')))
        self.assertEqual(actual_doc, expected_document)

    def test_deserialization_example_7(self):
        """
        Test the deserialization of example 7 which is a simple activity
        description.
        """
        actual_doc = prov.ProvDocument.deserialize(source=os.path.join(DATA_PATH, 'example_07.xml'), format='xml')
        expected_document = prov.ProvDocument()
        ex_ns = Namespace(*EX_NS)
        expected_document.add_namespace(ex_ns)
        expected_document.activity('ex:a1', '2011-11-16T16:05:00', '2011-11-16T16:06:00', [(prov.PROV_TYPE, QualifiedName(ex_ns, 'edit')), ('ex:host', 'server.example.org')])
        self.assertEqual(actual_doc, expected_document)

    def test_deserialization_example_04_and_05(self):
        """
        Example 4 and 5 have a different type specification. They use an
        xsi:type as an attribute on an entity. This can be read but if
        written again it will become an XML child element. This is
        semantically identical but cannot be tested with a round trip.
        """
        xml_string = '\n        <prov:document\n            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n            xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n            xmlns:prov="http://www.w3.org/ns/prov#"\n            xmlns:ex="http://example.com/ns/ex#"\n            xmlns:tr="http://example.com/ns/tr#">\n\n          <prov:entity prov:id="tr:WD-prov-dm-20111215" xsi:type="prov:Plan">\n            <prov:type xsi:type="xsd:QName">ex:Workflow</prov:type>\n          </prov:entity>\n\n        </prov:document>\n        '
        with io.StringIO() as xml:
            xml.write(xml_string)
            xml.seek(0, 0)
            actual_document = prov.ProvDocument.deserialize(source=xml, format='xml')
        expected_document = prov.ProvDocument()
        ex_ns = Namespace(*EX_NS)
        expected_document.add_namespace(ex_ns)
        expected_document.add_namespace(*EX_TR)
        expected_document.entity('tr:WD-prov-dm-20111215', ((prov.PROV_TYPE, QualifiedName(ex_ns, 'Workflow')), (prov.PROV_TYPE, PROV['Plan'])))
        self.assertEqual(actual_document, expected_document, 'example_04')
        xml_string = '\n        <prov:document\n          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n          xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n          xmlns:prov="http://www.w3.org/ns/prov#"\n          xmlns:ex="http://example.com/ns/ex#"\n          xmlns:tr="http://example.com/ns/tr#">\n\n        <prov:entity prov:id="tr:WD-prov-dm-20111215" xsi:type="prov:Plan">\n          <prov:type xsi:type="xsd:QName">ex:Workflow</prov:type>\n          <prov:type xsi:type="xsd:QName">prov:Plan</prov:type> <!-- inferred -->\n          <prov:type xsi:type="xsd:QName">prov:Entity</prov:type> <!-- inferred -->\n        </prov:entity>\n\n        </prov:document>\n        '
        with io.StringIO() as xml:
            xml.write(xml_string)
            xml.seek(0, 0)
            actual_document = prov.ProvDocument.deserialize(source=xml, format='xml')
        expected_document = prov.ProvDocument()
        expected_document.add_namespace(*EX_NS)
        expected_document.add_namespace(*EX_TR)
        expected_document.entity('tr:WD-prov-dm-20111215', ((prov.PROV_TYPE, QualifiedName(ex_ns, 'Workflow')), (prov.PROV_TYPE, PROV['Entity']), (prov.PROV_TYPE, PROV['Plan'])))
        self.assertEqual(actual_document, expected_document, 'example_05')

    def test_other_elements(self):
        """
        PROV XML uses the <prov:other> element to enable the storage of non
        PROV information in a PROV XML document. It will be ignored by this
        library a warning will be raised informing the user.
        """
        xml_string = '\n        <prov:document\n            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n            xmlns:xsd="http://www.w3.org/2001/XMLSchema"\n            xmlns:prov="http://www.w3.org/ns/prov#"\n            xmlns:ex="http://example.com/ns/ex#">\n\n          <!-- prov statements go here -->\n\n          <prov:other>\n            <ex:foo>\n              <ex:content>bar</ex:content>\n            </ex:foo>\n          </prov:other>\n\n          <!-- more prov statements can go here -->\n\n        </prov:document>\n        '
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with io.StringIO() as xml:
                xml.write(xml_string)
                xml.seek(0, 0)
                doc = prov.ProvDocument.deserialize(source=xml, format='xml')
        self.assertEqual(len(w), 1)
        self.assertTrue('Document contains non-PROV information in <prov:other>. It will be ignored in this package.' in str(w[0].message))
        self.assertEqual(len(doc._records), 0)

    def test_nested_default_namespace(self):
        """
        Tests that a default namespace that is defined in a lower level tag is
        written to a bundle.
        """
        filename = os.path.join(DATA_PATH, 'nested_default_namespace.xml')
        doc = prov.ProvDocument.deserialize(source=filename, format='xml')
        ns = Namespace('', 'http://example.org/0/')
        self.assertEqual(len(doc._records), 1)
        self.assertEqual(doc.get_default_namespace(), ns)
        self.assertEqual(doc._records[0].identifier.namespace, ns)
        self.assertEqual(doc._records[0].identifier.localpart, 'e001')

    def test_redefining_namespaces(self):
        """
        Test the behaviour when namespaces are redefined at the element level.
        """
        filename = os.path.join(DATA_PATH, 'namespace_redefined_but_does_not_change.xml')
        doc = prov.ProvDocument.deserialize(source=filename, format='xml')
        self.assertEqual(len(doc._records), 1)
        ns = Namespace('ex', 'http://example.com/ns/ex#')
        self.assertEqual(doc._records[0].attributes[0][1].namespace, ns)
        filename = os.path.join(DATA_PATH, 'namespace_redefined.xml')
        doc = prov.ProvDocument.deserialize(source=filename, format='xml')
        new_ns = doc._records[0].attributes[0][1].namespace
        self.assertNotEqual(new_ns, ns)
        self.assertEqual(new_ns.uri, 'http://example.com/ns/new_ex#')