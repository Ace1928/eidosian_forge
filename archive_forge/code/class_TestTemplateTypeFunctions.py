import json
import tempfile
from unittest import mock
import io
from oslo_serialization import base64
import testtools
from testtools import matchers
from urllib import error
import yaml
from heatclient.common import template_utils
from heatclient.common import utils
from heatclient import exc
class TestTemplateTypeFunctions(testtools.TestCase):
    hot_template = b'heat_template_version: 2013-05-23\nparameters:\n  param1:\n    type: string\nresources:\n  resource1:\n    type: foo.yaml\n    properties:\n      foo: bar\n  resource2:\n    type: OS::Heat::ResourceGroup\n    properties:\n      resource_def:\n        type: spam/egg.yaml\n    '
    foo_template = b'heat_template_version: "2013-05-23"\nparameters:\n  foo:\n    type: string\n    '
    egg_template = b'heat_template_version: "2013-05-23"\nparameters:\n  egg:\n    type: string\n    '

    @mock.patch('urllib.request.urlopen')
    def test_hot_template(self, mock_url):
        tmpl_file = '/home/my/dir/template.yaml'
        url = 'file:///home/my/dir/template.yaml'

        def side_effect(args):
            if url == args:
                return io.BytesIO(self.hot_template)
            if 'file:///home/my/dir/foo.yaml' == args:
                return io.BytesIO(self.foo_template)
            if 'file:///home/my/dir/spam/egg.yaml' == args:
                return io.BytesIO(self.egg_template)
        mock_url.side_effect = side_effect
        files, tmpl_parsed = template_utils.get_template_contents(template_file=tmpl_file)
        self.assertEqual(yaml.safe_load(self.foo_template.decode('utf-8')), json.loads(files.get('file:///home/my/dir/foo.yaml')))
        self.assertEqual(yaml.safe_load(self.egg_template.decode('utf-8')), json.loads(files.get('file:///home/my/dir/spam/egg.yaml')))
        self.assertEqual({'heat_template_version': '2013-05-23', 'parameters': {'param1': {'type': 'string'}}, 'resources': {'resource1': {'type': 'file:///home/my/dir/foo.yaml', 'properties': {'foo': 'bar'}}, 'resource2': {'type': 'OS::Heat::ResourceGroup', 'properties': {'resource_def': {'type': 'file:///home/my/dir/spam/egg.yaml'}}}}}, tmpl_parsed)
        mock_url.assert_has_calls([mock.call('file:///home/my/dir/foo.yaml'), mock.call(url), mock.call('file:///home/my/dir/spam/egg.yaml')], any_order=True)