from twisted.trial import unittest
from twisted.words.protocols.jabber import error
from twisted.words.xish import domish
class ParseErrorTests(unittest.TestCase):
    """
    Tests for L{error._parseError}.
    """

    def setUp(self) -> None:
        self.error = domish.Element((None, 'error'))

    def test_empty(self) -> None:
        """
        Test parsing of the empty error element.
        """
        result = error._parseError(self.error, 'errorns')
        self.assertEqual({'condition': None, 'text': None, 'textLang': None, 'appCondition': None}, result)

    def test_condition(self) -> None:
        """
        Test parsing of an error element with a condition.
        """
        self.error.addElement(('errorns', 'bad-request'))
        result = error._parseError(self.error, 'errorns')
        self.assertEqual('bad-request', result['condition'])

    def test_text(self) -> None:
        """
        Test parsing of an error element with a text.
        """
        text = self.error.addElement(('errorns', 'text'))
        text.addContent('test')
        result = error._parseError(self.error, 'errorns')
        self.assertEqual('test', result['text'])
        self.assertEqual(None, result['textLang'])

    def test_textLang(self) -> None:
        """
        Test parsing of an error element with a text with a defined language.
        """
        text = self.error.addElement(('errorns', 'text'))
        text[NS_XML, 'lang'] = 'en_US'
        text.addContent('test')
        result = error._parseError(self.error, 'errorns')
        self.assertEqual('en_US', result['textLang'])

    def test_appCondition(self) -> None:
        """
        Test parsing of an error element with an app specific condition.
        """
        condition = self.error.addElement(('testns', 'condition'))
        result = error._parseError(self.error, 'errorns')
        self.assertEqual(condition, result['appCondition'])

    def test_appConditionMultiple(self) -> None:
        """
        Test parsing of an error element with multiple app specific conditions.
        """
        self.error.addElement(('testns', 'condition'))
        condition = self.error.addElement(('testns', 'condition2'))
        result = error._parseError(self.error, 'errorns')
        self.assertEqual(condition, result['appCondition'])