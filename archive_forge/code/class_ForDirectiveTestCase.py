import doctest
import re
import sys
import unittest
from genshi.compat import IS_PYTHON2
from genshi.template import directives, MarkupTemplate, TextTemplate, \
class ForDirectiveTestCase(unittest.TestCase):
    """Tests for the `py:for` template directive."""

    def test_loop_with_strip(self):
        """
        Verify that the combining the `py:for` directive with `py:strip` works
        correctly.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <div py:for="item in items" py:strip="">\n            <b>${item}</b>\n          </div>\n        </doc>')
        self.assertEqual('<doc>\n            <b>1</b>\n            <b>2</b>\n            <b>3</b>\n            <b>4</b>\n            <b>5</b>\n        </doc>', tmpl.generate(items=range(1, 6)).render(encoding=None))

    def test_as_element(self):
        """
        Verify that the directive can also be used as an element.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="item in items">\n            <b>${item}</b>\n          </py:for>\n        </doc>')
        self.assertEqual('<doc>\n            <b>1</b>\n            <b>2</b>\n            <b>3</b>\n            <b>4</b>\n            <b>5</b>\n        </doc>', tmpl.generate(items=range(1, 6)).render(encoding=None))

    def test_multi_assignment(self):
        """
        Verify that assignment to tuples works correctly.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="k, v in items">\n            <p>key=$k, value=$v</p>\n          </py:for>\n        </doc>')
        self.assertEqual('<doc>\n            <p>key=a, value=1</p>\n            <p>key=b, value=2</p>\n        </doc>', tmpl.generate(items=(('a', 1), ('b', 2))).render(encoding=None))

    def test_nested_assignment(self):
        """
        Verify that assignment to nested tuples works correctly.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="idx, (k, v) in items">\n            <p>$idx: key=$k, value=$v</p>\n          </py:for>\n        </doc>')
        self.assertEqual('<doc>\n            <p>0: key=a, value=1</p>\n            <p>1: key=b, value=2</p>\n        </doc>', tmpl.generate(items=enumerate([('a', 1), ('b', 2)])).render(encoding=None))

    def test_not_iterable(self):
        """
        Verify that assignment to nested tuples works correctly.
        """
        tmpl = MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n          <py:for each="item in foo">\n            $item\n          </py:for>\n        </doc>', filename='test.html')
        try:
            list(tmpl.generate(foo=12))
            self.fail('Expected TemplateRuntimeError')
        except TypeError as e:
            assert str(e) == 'iteration over non-sequence' or str(e) == "'int' object is not iterable"
            exc_type, exc_value, exc_traceback = sys.exc_info()
            frame = exc_traceback.tb_next
            frames = []
            while frame.tb_next:
                frame = frame.tb_next
                frames.append(frame)
            expected_iter_str = "u'iter(foo)'" if IS_PYTHON2 else "'iter(foo)'"
            self.assertEqual('<Expression %s>' % expected_iter_str, frames[-1].tb_frame.f_code.co_name)
            self.assertEqual('test.html', frames[-1].tb_frame.f_code.co_filename)
            self.assertEqual(2, frames[-1].tb_lineno)

    def test_for_with_empty_value(self):
        """
        Verify an empty 'for' value is an error
        """
        try:
            MarkupTemplate('<doc xmlns:py="http://genshi.edgewall.org/">\n              <py:for each="">\n                empty\n              </py:for>\n            </doc>', filename='test.html').generate()
            self.fail('ExpectedTemplateSyntaxError')
        except TemplateSyntaxError as e:
            self.assertEqual('test.html', e.filename)
            if sys.version_info[:2] > (2, 4):
                self.assertEqual(2, e.lineno)