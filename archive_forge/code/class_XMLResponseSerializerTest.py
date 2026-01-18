import collections
import datetime
from lxml import etree
from oslo_serialization import jsonutils as json
import webob
from heat.common import serializers
from heat.tests import common
class XMLResponseSerializerTest(common.HeatTestCase):

    def _recursive_dict(self, element):
        return (element.tag, dict(map(self._recursive_dict, element)) or element.text)

    def test_to_xml(self):
        fixture = {'key': 'value'}
        expected = b'<key>value</key>'
        actual = serializers.XMLResponseSerializer().to_xml(fixture)
        self.assertEqual(expected, actual)

    def test_to_xml_with_date_format_value(self):
        fixture = {'date': datetime.datetime(1, 3, 8, 2)}
        expected = b'<date>0001-03-08 02:00:00</date>'
        actual = serializers.XMLResponseSerializer().to_xml(fixture)
        self.assertEqual(expected, actual)

    def test_to_xml_with_list(self):
        fixture = {'name': ['1', '2']}
        expected = b'<name><member>1</member><member>2</member></name>'
        actual = serializers.XMLResponseSerializer().to_xml(fixture)
        actual_xml_tree = etree.XML(actual)
        actual_xml_dict = self._recursive_dict(actual_xml_tree)
        expected_xml_tree = etree.XML(expected)
        expected_xml_dict = self._recursive_dict(expected_xml_tree)
        self.assertEqual(expected_xml_dict, actual_xml_dict)

    def test_to_xml_with_more_deep_format(self):
        fixture = collections.OrderedDict([('aresponse', collections.OrderedDict([('is_public', True), ('name', [collections.OrderedDict([('name1', 'test')])])]))])
        expected = '<aresponse><is_public>True</is_public><name><member><name1>test</name1></member></name></aresponse>'.encode('latin-1')
        actual = serializers.XMLResponseSerializer().to_xml(fixture)
        actual_xml_tree = etree.XML(actual)
        actual_xml_dict = self._recursive_dict(actual_xml_tree)
        expected_xml_tree = etree.XML(expected)
        expected_xml_dict = self._recursive_dict(expected_xml_tree)
        self.assertEqual(expected_xml_dict, actual_xml_dict)

    def test_to_xml_with_json_only_keys(self):
        fixture = collections.OrderedDict([('aresponse', collections.OrderedDict([('is_public', True), ('TemplateBody', {'name1': 'test'}), ('Metadata', {'name2': 'test2'})]))])
        expected = '<aresponse><is_public>True</is_public><TemplateBody>{"name1": "test"}</TemplateBody><Metadata>{"name2": "test2"}</Metadata></aresponse>'.encode('latin-1')
        actual = serializers.XMLResponseSerializer().to_xml(fixture)
        actual_xml_tree = etree.XML(actual)
        actual_xml_dict = self._recursive_dict(actual_xml_tree)
        expected_xml_tree = etree.XML(expected)
        expected_xml_dict = self._recursive_dict(expected_xml_tree)
        self.assertEqual(expected_xml_dict, actual_xml_dict)

    def test_default(self):
        fixture = {'key': 'value'}
        response = webob.Response()
        serializers.XMLResponseSerializer().default(response, fixture)
        self.assertEqual(200, response.status_int)
        content_types = list(filter(lambda h: h[0] == 'Content-Type', response.headerlist))
        self.assertEqual(1, len(content_types))
        self.assertEqual('application/xml', response.content_type)
        self.assertEqual(b'<key>value</key>', response.body)