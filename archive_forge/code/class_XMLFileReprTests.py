import sys
from io import StringIO
from typing import List, Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, succeed
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.trial.util import suppress as SUPPRESS
from twisted.web._element import UnexposedMethodError
from twisted.web.error import FlattenerError, MissingRenderMethod, MissingTemplateLoader
from twisted.web.iweb import IRequest, ITemplateLoader
from twisted.web.server import NOT_DONE_YET
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
from twisted.web.test.test_web import DummyRequest
class XMLFileReprTests(TestCase):
    """
    Tests for L{twisted.web.template.XMLFile}'s C{__repr__}.
    """

    def test_filePath(self) -> None:
        """
        An L{XMLFile} with a L{FilePath} returns a useful repr().
        """
        path = FilePath('/tmp/fake.xml')
        self.assertEqual(f'<XMLFile of {path!r}>', repr(XMLFile(path)))

    def test_filename(self) -> None:
        """
        An L{XMLFile} with a filename returns a useful repr().
        """
        fname = '/tmp/fake.xml'
        self.assertEqual(f'<XMLFile of {fname!r}>', repr(XMLFile(fname)))
    test_filename.suppress = [_xmlFileSuppress]

    def test_file(self) -> None:
        """
        An L{XMLFile} with a file object returns a useful repr().
        """
        fobj = StringIO('not xml')
        self.assertEqual(f'<XMLFile of {fobj!r}>', repr(XMLFile(fobj)))
    test_file.suppress = [_xmlFileSuppress]