from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
class BrokenHTMLTests(TestCase):
    """
    Tests for when microdom encounters very bad HTML and C{beExtremelyLenient}
    is enabled. These tests are inspired by some HTML generated in by a mailer,
    which breaks up very long lines by splitting them with '!\\n '.
    The expected behaviour is loosely modelled on the way Firefox treats very
    bad HTML.
    """

    def checkParsed(self, input: str, expected: str, beExtremelyLenient: int=1) -> None:
        """
        Check that C{input}, when parsed, produces a DOM where the XML
        of the document element is equal to C{expected}.
        """
        output = microdom.parseString(input, beExtremelyLenient=beExtremelyLenient)
        self.assertEqual(output.documentElement.toxml(), expected)

    def test_brokenAttributeName(self) -> None:
        """
        Check that microdom does its best to handle broken attribute names.
        The important thing is that it doesn't raise an exception.
        """
        input = '<body><h1><div al!\n ign="center">Foo</div></h1></body>'
        expected = '<body><h1><div al="True" ign="center">Foo</div></h1></body>'
        self.checkParsed(input, expected)

    def test_brokenAttributeValue(self) -> None:
        """
        Check that microdom encompasses broken attribute values.
        """
        input = '<body><h1><div align="cen!\n ter">Foo</div></h1></body>'
        expected = '<body><h1><div align="cen!\n ter">Foo</div></h1></body>'
        self.checkParsed(input, expected)

    def test_brokenOpeningTag(self) -> None:
        """
        Check that microdom does its best to handle broken opening tags.
        The important thing is that it doesn't raise an exception.
        """
        input = '<body><h1><sp!\n an>Hello World!</span></h1></body>'
        expected = '<body><h1><sp an="True">Hello World!</sp></h1></body>'
        self.checkParsed(input, expected)

    def test_brokenSelfClosingTag(self) -> None:
        """
        Check that microdom does its best to handle broken self-closing tags
        The important thing is that it doesn't raise an exception.
        """
        self.checkParsed('<body><span /!\n></body>', '<body><span></span></body>')
        self.checkParsed('<span!\n />', '<span></span>')

    def test_brokenClosingTag(self) -> None:
        """
        Check that microdom does its best to handle broken closing tags.
        The important thing is that it doesn't raise an exception.
        """
        input = '<body><h1><span>Hello World!</sp!\nan></h1></body>'
        expected = '<body><h1><span>Hello World!</span></h1></body>'
        self.checkParsed(input, expected)
        input = '<body><h1><span>Hello World!</!\nspan></h1></body>'
        self.checkParsed(input, expected)
        input = '<body><h1><span>Hello World!</span!\n></h1></body>'
        self.checkParsed(input, expected)
        input = '<body><h1><span>Hello World!<!\n/span></h1></body>'
        expected = '<body><h1><span>Hello World!<!></!></span></h1></body>'
        self.checkParsed(input, expected)