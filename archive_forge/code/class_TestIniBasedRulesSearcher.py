import sys
from breezy import rules, tests
class TestIniBasedRulesSearcher(tests.TestCase):

    def make_searcher(self, text):
        """Make a _RulesSearcher from a string"""
        if text is None:
            lines = None
        else:
            lines = text.splitlines()
        return rules._IniBasedRulesSearcher(lines)

    def test_unknown_namespace(self):
        self.assertRaises(rules.UnknownRules, rules._IniBasedRulesSearcher, ['[junk]', 'foo=bar'])

    def test_get_items_file_missing(self):
        rs = self.make_searcher(None)
        self.assertEqual((), rs.get_items('a.txt'))
        self.assertEqual((), rs.get_selected_items('a.txt', ['foo']))
        self.assertEqual(None, rs.get_single_value('a.txt', 'foo'))

    def test_get_items_file_empty(self):
        rs = self.make_searcher('')
        self.assertEqual((), rs.get_items('a.txt'))
        self.assertEqual((), rs.get_selected_items('a.txt', ['foo']))
        self.assertEqual(None, rs.get_single_value('a.txt', 'foo'))

    def test_get_items_from_extension_match(self):
        rs = self.make_searcher('[name *.txt]\nfoo=bar\na=True\n')
        self.assertEqual((), rs.get_items('a.py'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('a.txt'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('dir/a.txt'))
        self.assertEqual((('foo', 'bar'),), rs.get_selected_items('a.txt', ['foo']))
        self.assertEqual('bar', rs.get_single_value('a.txt', 'foo'))

    def test_get_items_from_multiple_glob_match(self):
        rs = self.make_searcher('[name *.txt *.py \'x x\' "y y"]\nfoo=bar\na=True\n')
        self.assertEqual((), rs.get_items('NEWS'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('a.py'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('a.txt'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('x x'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('y y'))
        self.assertEqual('bar', rs.get_single_value('a.txt', 'foo'))

    def test_get_items_pathname_match(self):
        rs = self.make_searcher('[name ./a.txt]\nfoo=baz\n')
        self.assertEqual((('foo', 'baz'),), rs.get_items('a.txt'))
        self.assertEqual('baz', rs.get_single_value('a.txt', 'foo'))
        self.assertEqual((), rs.get_items('dir/a.txt'))
        self.assertEqual(None, rs.get_single_value('dir/a.txt', 'foo'))

    def test_get_items_match_first(self):
        rs = self.make_searcher('[name ./a.txt]\nfoo=baz\n[name *.txt]\nfoo=bar\na=True\n')
        self.assertEqual((('foo', 'baz'),), rs.get_items('a.txt'))
        self.assertEqual('baz', rs.get_single_value('a.txt', 'foo'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('dir/a.txt'))
        self.assertEqual('bar', rs.get_single_value('dir/a.txt', 'foo'))