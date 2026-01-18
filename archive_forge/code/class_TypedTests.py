from twisted.python import usage
from twisted.trial import unittest
class TypedTests(unittest.TestCase):
    """
    Test L{usage.Options.parseOptions} for options with forced types.
    """

    def setUp(self):
        self.usage = TypedOptions()

    def test_defaultValues(self):
        """
        Default values are parsed.
        """
        argV = []
        self.usage.parseOptions(argV)
        self.assertEqual(self.usage.opts['fooint'], 392)
        self.assertIsInstance(self.usage.opts['fooint'], int)
        self.assertEqual(self.usage.opts['foofloat'], 4.23)
        self.assertIsInstance(self.usage.opts['foofloat'], float)
        self.assertIsNone(self.usage.opts['eggint'])
        self.assertIsNone(self.usage.opts['eggfloat'])

    def test_parsingValues(self):
        """
        int and float values are parsed.
        """
        argV = '--fooint 912 --foofloat -823.1 --eggint 32 --eggfloat 21'.split()
        self.usage.parseOptions(argV)
        self.assertEqual(self.usage.opts['fooint'], 912)
        self.assertIsInstance(self.usage.opts['fooint'], int)
        self.assertEqual(self.usage.opts['foofloat'], -823.1)
        self.assertIsInstance(self.usage.opts['foofloat'], float)
        self.assertEqual(self.usage.opts['eggint'], 32)
        self.assertIsInstance(self.usage.opts['eggint'], int)
        self.assertEqual(self.usage.opts['eggfloat'], 21.0)
        self.assertIsInstance(self.usage.opts['eggfloat'], float)

    def test_underscoreOption(self):
        """
        A dash in an option name is translated to an underscore before being
        dispatched to a handler.
        """
        self.usage.parseOptions(['--under-score', 'foo'])
        self.assertEqual(self.usage.underscoreValue, 'foo')

    def test_underscoreOptionAlias(self):
        """
        An option name with a dash in it can have an alias.
        """
        self.usage.parseOptions(['-u', 'bar'])
        self.assertEqual(self.usage.underscoreValue, 'bar')

    def test_invalidValues(self):
        """
        Passing wrong values raises an error.
        """
        argV = '--fooint egg'.split()
        self.assertRaises(usage.UsageError, self.usage.parseOptions, argV)