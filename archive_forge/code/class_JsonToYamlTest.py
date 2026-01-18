import os
from unittest import mock
import re
import yaml
from heat.common import config
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.tests import common
from heat.tests import utils
class JsonToYamlTest(common.HeatTestCase):

    def setUp(self):
        super(JsonToYamlTest, self).setUp()
        self.expected_test_count = 2
        self.longMessage = True
        self.maxDiff = None

    def test_convert_all_templates(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates')
        template_test_count = 0
        for json_str, yml_str in self.convert_all_json_to_yaml(path):
            self.compare_json_vs_yaml(json_str, yml_str)
            template_test_count += 1
            if template_test_count >= self.expected_test_count:
                break
        self.assertGreaterEqual(template_test_count, self.expected_test_count, 'Expected at least %d templates to be tested, not %d' % (self.expected_test_count, template_test_count))

    def compare_json_vs_yaml(self, json_str, yml_str):
        yml = template_format.parse(yml_str)
        self.assertEqual(u'2012-12-12', yml[u'HeatTemplateFormatVersion'])
        self.assertNotIn(u'AWSTemplateFormatVersion', yml)
        del yml[u'HeatTemplateFormatVersion']
        jsn = template_format.parse(json_str)
        if u'AWSTemplateFormatVersion' in jsn:
            del jsn[u'AWSTemplateFormatVersion']
        self.assertEqual(yml, jsn)

    def convert_all_json_to_yaml(self, dirpath):
        for path in os.listdir(dirpath):
            if not path.endswith('.template') and (not path.endswith('.json')):
                continue
            with open(os.path.join(dirpath, path), 'r') as f:
                json_str = f.read()
            yml_str = template_format.convert_json_to_yaml(json_str)
            yield (json_str, yml_str)

    def test_integer_only_keys_get_translated_correctly(self):
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates/WordPress_Single_Instance.template')
        with open(path, 'r') as f:
            json_str = f.read()
            yml_str = template_format.convert_json_to_yaml(json_str)
            match = re.search('[\\s,{]\\d+\\s*:', yml_str)
            self.assertIsNone(match)