import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
class MarkupTemplateEnginePluginTestCase(unittest.TestCase):

    def test_init_no_options(self):
        plugin = MarkupTemplateEnginePlugin()
        self.assertEqual(None, plugin.default_encoding)
        self.assertEqual('html', plugin.default_format)
        self.assertEqual(None, plugin.default_doctype)
        self.assertEqual([], plugin.loader.search_path)
        self.assertEqual(True, plugin.loader.auto_reload)
        self.assertEqual(25, plugin.loader._cache.capacity)

    def test_init_with_loader_options(self):
        plugin = MarkupTemplateEnginePlugin(options={'genshi.auto_reload': 'off', 'genshi.max_cache_size': '100', 'genshi.search_path': '/usr/share/tmpl:/usr/local/share/tmpl'})
        self.assertEqual(['/usr/share/tmpl', '/usr/local/share/tmpl'], plugin.loader.search_path)
        self.assertEqual(False, plugin.loader.auto_reload)
        self.assertEqual(100, plugin.loader._cache.capacity)

    def test_init_with_invalid_cache_size(self):
        self.assertRaises(ConfigurationError, MarkupTemplateEnginePlugin, options={'genshi.max_cache_size': 'thirty'})

    def test_init_with_output_options(self):
        plugin = MarkupTemplateEnginePlugin(options={'genshi.default_encoding': 'iso-8859-15', 'genshi.default_format': 'xhtml', 'genshi.default_doctype': 'xhtml-strict'})
        self.assertEqual('iso-8859-15', plugin.default_encoding)
        self.assertEqual('xhtml', plugin.default_format)
        self.assertEqual(DocType.XHTML, plugin.default_doctype)

    def test_init_with_invalid_output_format(self):
        self.assertRaises(ConfigurationError, MarkupTemplateEnginePlugin, options={'genshi.default_format': 'foobar'})

    def test_init_with_invalid_doctype(self):
        self.assertRaises(ConfigurationError, MarkupTemplateEnginePlugin, options={'genshi.default_doctype': 'foobar'})

    def test_load_template_from_file(self):
        plugin = MarkupTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        self.assertEqual('test.html', os.path.basename(tmpl.filename))
        assert isinstance(tmpl, MarkupTemplate)

    def test_load_template_from_string(self):
        plugin = MarkupTemplateEnginePlugin()
        tmpl = plugin.load_template(None, template_string='<p>\n          $message\n        </p>')
        self.assertEqual(None, tmpl.filename)
        assert isinstance(tmpl, MarkupTemplate)

    def test_transform_with_load(self):
        plugin = MarkupTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        stream = plugin.transform({'message': 'Hello'}, tmpl)
        assert isinstance(stream, Stream)

    def test_transform_without_load(self):
        plugin = MarkupTemplateEnginePlugin()
        stream = plugin.transform({'message': 'Hello'}, PACKAGE + '.templates.test')
        assert isinstance(stream, Stream)

    def test_render(self):
        plugin = MarkupTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        output = plugin.render({'message': 'Hello'}, template=tmpl)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n<html lang="en">\n  <head>\n    <title>Test</title>\n  </head>\n  <body>\n    <h1>Test</h1>\n    <p>Hello</p>\n  </body>\n</html>', output)

    def test_render_with_format(self):
        plugin = MarkupTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        output = plugin.render({'message': 'Hello'}, format='xhtml', template=tmpl)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">\n<html xmlns="http://www.w3.org/1999/xhtml" lang="en">\n  <head>\n    <title>Test</title>\n  </head>\n  <body>\n    <h1>Test</h1>\n    <p>Hello</p>\n  </body>\n</html>', output)

    def test_render_with_doctype(self):
        plugin = MarkupTemplateEnginePlugin(options={'genshi.default_doctype': 'html-strict'})
        tmpl = plugin.load_template(PACKAGE + '.templates.test')
        output = plugin.render({'message': 'Hello'}, template=tmpl)
        self.assertEqual('<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd">\n<html lang="en">\n  <head>\n    <title>Test</title>\n  </head>\n  <body>\n    <h1>Test</h1>\n    <p>Hello</p>\n  </body>\n</html>', output)

    def test_render_fragment_with_doctype(self):
        plugin = MarkupTemplateEnginePlugin(options={'genshi.default_doctype': 'html-strict'})
        tmpl = plugin.load_template(PACKAGE + '.templates.test_no_doctype')
        output = plugin.render({'message': 'Hello'}, template=tmpl, fragment=True)
        self.assertEqual('<html lang="en">\n  <head>\n    <title>Test</title>\n  </head>\n  <body>\n    <h1>Test</h1>\n    <p>Hello</p>\n  </body>\n</html>', output)

    def test_helper_functions(self):
        plugin = MarkupTemplateEnginePlugin()
        tmpl = plugin.load_template(PACKAGE + '.templates.functions')
        output = plugin.render({'snippet': u'<b>Foo</b>'}, template=tmpl)
        self.assertEqual('<div>\nFalse\nbar\n<b>Foo</b>\n<b>Foo</b>\n</div>', output)