import unittest
from Cython.Compiler import Code, UtilityCode
class TestTempitaUtilityLoader(TestUtilityLoader):
    """
    Test loading UtilityCodes with Tempita substitution
    """
    expected_tempita = (TestUtilityLoader.expected[0].replace('{{loader}}', 'Loader'), TestUtilityLoader.expected[1].replace('{{loader}}', 'Loader'))
    required_tempita = (TestUtilityLoader.required[0].replace('{{loader}}', 'Loader'), TestUtilityLoader.required[1].replace('{{loader}}', 'Loader'))
    cls = Code.TempitaUtilityCode

    def test_load_as_string(self):
        got = strip_2tup(self.cls.load_as_string(self.name, self.filename, context=self.context))
        self.assertEqual(got, self.expected_tempita)

    def test_load(self):
        utility = self.cls.load(self.name, self.filename, context=self.context)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected_tempita)
        required, = utility.requires
        got = strip_2tup((required.proto, required.impl))
        self.assertEqual(got, self.required_tempita)
        utility = self.cls.load(self.name, from_file=self.filename, context=self.context)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected_tempita)