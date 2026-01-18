import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
class CallablesInConfigTest(unittest.TestCase):
    setup_server = staticmethod(setup_server)

    def test_call_with_literal_dict(self):
        from textwrap import dedent
        conf = dedent("\n        [my]\n        value = dict(**{'foo': 'bar'})\n        ")
        fp = StringIOFromNative(conf)
        cherrypy.config.update(fp)
        self.assertEqual(cherrypy.config['my']['value'], {'foo': 'bar'})

    def test_call_with_kwargs(self):
        from textwrap import dedent
        conf = dedent('\n        [my]\n        value = dict(foo="buzz", **cherrypy._test_dict)\n        ')
        test_dict = {'foo': 'bar', 'bar': 'foo', 'fizz': 'buzz'}
        cherrypy._test_dict = test_dict
        fp = StringIOFromNative(conf)
        cherrypy.config.update(fp)
        test_dict['foo'] = 'buzz'
        self.assertEqual(cherrypy.config['my']['value']['foo'], 'buzz')
        self.assertEqual(cherrypy.config['my']['value'], test_dict)
        del cherrypy._test_dict