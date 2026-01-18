import io
import os
import tempfile
import unittest
from testtools import TestCase
from testtools.compat import (
from testtools.content import (
from testtools.content_type import (
from testtools.matchers import (
from testtools.tests.helpers import an_exc_info
class TestAttachFile(TestCase):

    def make_file(self, data):
        fd, path = tempfile.mkstemp()
        self.addCleanup(os.remove, path)
        os.write(fd, _b(data))
        os.close(fd)
        return path

    def test_simple(self):

        class SomeTest(TestCase):

            def test_foo(self):
                pass
        test = SomeTest('test_foo')
        data = 'some data'
        path = self.make_file(data)
        my_content = text_content(data)
        attach_file(test, path, name='foo')
        self.assertEqual({'foo': my_content}, test.getDetails())

    def test_optional_name(self):

        class SomeTest(TestCase):

            def test_foo(self):
                pass
        test = SomeTest('test_foo')
        path = self.make_file('some data')
        base_path = os.path.basename(path)
        attach_file(test, path)
        self.assertEqual([base_path], list(test.getDetails()))

    def test_lazy_read(self):

        class SomeTest(TestCase):

            def test_foo(self):
                pass
        test = SomeTest('test_foo')
        path = self.make_file('some data')
        attach_file(test, path, name='foo', buffer_now=False)
        content = test.getDetails()['foo']
        content_file = open(path, 'w')
        content_file.write('new data')
        content_file.close()
        self.assertEqual(''.join(content.iter_text()), 'new data')

    def test_eager_read_by_default(self):

        class SomeTest(TestCase):

            def test_foo(self):
                pass
        test = SomeTest('test_foo')
        path = self.make_file('some data')
        attach_file(test, path, name='foo')
        content = test.getDetails()['foo']
        content_file = open(path, 'w')
        content_file.write('new data')
        content_file.close()
        self.assertEqual(''.join(content.iter_text()), 'some data')