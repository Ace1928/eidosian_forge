import os
import tempfile
import unittest
from webtest import TestApp
import pecan
from pecan.tests import PecanTestCase
class TestGlobalConfig(PecanTestCase):

    def tearDown(self):
        from pecan import configuration
        configuration.set_config(dict(configuration.initconf()), overwrite=True)

    def test_paint_from_dict(self):
        from pecan import configuration
        configuration.set_config({'foo': 'bar'})
        assert dict(configuration._runtime_conf) != {'foo': 'bar'}
        self.assertEqual(configuration._runtime_conf.foo, 'bar')

    def test_overwrite_from_dict(self):
        from pecan import configuration
        configuration.set_config({'foo': 'bar'}, overwrite=True)
        assert dict(configuration._runtime_conf) == {'foo': 'bar'}

    def test_paint_from_file(self):
        from pecan import configuration
        configuration.set_config(os.path.join(__here__, 'config_fixtures/foobar.py'))
        assert dict(configuration._runtime_conf) != {'foo': 'bar'}
        assert configuration._runtime_conf.foo == 'bar'

    def test_overwrite_from_file(self):
        from pecan import configuration
        configuration.set_config(os.path.join(__here__, 'config_fixtures/foobar.py'), overwrite=True)
        assert dict(configuration._runtime_conf) == {'foo': 'bar'}

    def test_set_config_none_type(self):
        from pecan import configuration
        self.assertRaises(RuntimeError, configuration.set_config, None)

    def test_set_config_to_dir(self):
        from pecan import configuration
        self.assertRaises(RuntimeError, configuration.set_config, '/')