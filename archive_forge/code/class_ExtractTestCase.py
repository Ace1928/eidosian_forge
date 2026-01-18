from datetime import datetime
from gettext import NullTranslations
import unittest
import six
from genshi.core import Attrs
from genshi.template import MarkupTemplate, Context
from genshi.filters.i18n import Translator, extract
from genshi.input import HTML
from genshi.compat import IS_PYTHON2, StringIO
from genshi.tests.test_utils import doctest_suite
class ExtractTestCase(unittest.TestCase):

    def test_markup_template_extraction(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <head>\n            <title>Example</title>\n          </head>\n          <body>\n            <h1>Example</h1>\n            <p>${_("Hello, %(name)s") % dict(name=username)}</p>\n            <p>${ngettext("You have %d item", "You have %d items", num)}</p>\n          </body>\n        </html>')
        results = list(extract(buf, ['_', 'ngettext'], [], {}))
        self.assertEqual([(3, None, 'Example', []), (6, None, 'Example', []), (7, '_', 'Hello, %(name)s', []), (8, 'ngettext', ('You have %d item', 'You have %d items', None), [])], results)

    def test_extraction_without_text(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <p title="Bar">Foo</p>\n          ${ngettext("Singular", "Plural", num)}\n        </html>')
        results = list(extract(buf, ['_', 'ngettext'], [], {'extract_text': 'no'}))
        self.assertEqual([(3, 'ngettext', ('Singular', 'Plural', None), [])], results)

    def test_text_template_extraction(self):
        buf = StringIO('${_("Dear %(name)s") % {\'name\': name}},\n\n        ${ngettext("Your item:", "Your items", len(items))}\n        #for item in items\n         * $item\n        #end\n\n        All the best,\n        Foobar')
        results = list(extract(buf, ['_', 'ngettext'], [], {'template_class': 'genshi.template:TextTemplate'}))
        self.assertEqual([(1, '_', 'Dear %(name)s', []), (3, 'ngettext', ('Your item:', 'Your items', None), []), (7, None, 'All the best,\n        Foobar', [])], results)

    def test_extraction_with_keyword_arg(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          ${gettext(\'Foobar\', foo=\'bar\')}\n        </html>')
        results = list(extract(buf, ['gettext'], [], {}))
        self.assertEqual([(2, 'gettext', 'Foobar', [])], results)

    def test_extraction_with_nonstring_arg(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          ${dgettext(curdomain, \'Foobar\')}\n        </html>')
        results = list(extract(buf, ['dgettext'], [], {}))
        self.assertEqual([(2, 'dgettext', (None, 'Foobar'), [])], results)

    def test_extraction_inside_ignored_tags(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <script type="text/javascript">\n            $(\'#llist\').tabs({\n              remote: true,\n              spinner: "${_(\'Please wait...\')}"\n            });\n          </script>\n        </html>')
        results = list(extract(buf, ['_'], [], {}))
        self.assertEqual([(5, '_', 'Please wait...', [])], results)

    def test_extraction_inside_ignored_tags_with_directives(self):
        buf = StringIO('<html xmlns:py="http://genshi.edgewall.org/">\n          <script type="text/javascript">\n            <py:if test="foobar">\n              alert("This shouldn\'t be extracted");\n            </py:if>\n          </script>\n        </html>')
        self.assertEqual([], list(extract(buf, ['_'], [], {})))

    def test_extract_py_def_directive_with_py_strip(self):
        tmpl = MarkupTemplate('<html xmlns:py="http://genshi.edgewall.org/" py:strip="">\n    <py:def function="diff_options_fields(diff)">\n    <label for="style">View differences</label>\n    <select id="style" name="style">\n      <option selected="${diff.style == \'inline\' or None}"\n              value="inline">inline</option>\n      <option selected="${diff.style == \'sidebyside\' or None}"\n              value="sidebyside">side by side</option>\n    </select>\n    <div class="field">\n      Show <input type="text" name="contextlines" id="contextlines" size="2"\n                  maxlength="3" value="${diff.options.contextlines &lt; 0 and \'all\' or diff.options.contextlines}" />\n      <label for="contextlines">lines around each change</label>\n    </div>\n    <fieldset id="ignore" py:with="options = diff.options">\n      <legend>Ignore:</legend>\n      <div class="field">\n        <input type="checkbox" id="ignoreblanklines" name="ignoreblanklines"\n               checked="${options.ignoreblanklines or None}" />\n        <label for="ignoreblanklines">Blank lines</label>\n      </div>\n      <div class="field">\n        <input type="checkbox" id="ignorecase" name="ignorecase"\n               checked="${options.ignorecase or None}" />\n        <label for="ignorecase">Case changes</label>\n      </div>\n      <div class="field">\n        <input type="checkbox" id="ignorewhitespace" name="ignorewhitespace"\n               checked="${options.ignorewhitespace or None}" />\n        <label for="ignorewhitespace">White space changes</label>\n      </div>\n    </fieldset>\n    <div class="buttons">\n      <input type="submit" name="update" value="${_(\'Update\')}" />\n    </div>\n  </py:def></html>')
        translator = Translator()
        tmpl.add_directives(Translator.NAMESPACE, translator)
        messages = list(translator.extract(tmpl.stream))
        self.assertEqual(10, len(messages))
        self.assertEqual([(3, None, 'View differences', []), (6, None, 'inline', []), (8, None, 'side by side', []), (10, None, 'Show', []), (13, None, 'lines around each change', []), (16, None, 'Ignore:', []), (20, None, 'Blank lines', []), (25, None, 'Case changes', []), (30, None, 'White space changes', []), (34, '_', 'Update', [])], messages)