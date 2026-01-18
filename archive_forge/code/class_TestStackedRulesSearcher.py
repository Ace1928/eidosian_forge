import sys
from breezy import rules, tests
class TestStackedRulesSearcher(tests.TestCase):

    def make_searcher(self, text1=None, text2=None):
        """Make a _StackedRulesSearcher with 0, 1 or 2 items"""
        searchers = []
        if text1 is not None:
            searchers.append(rules._IniBasedRulesSearcher(text1.splitlines()))
        if text2 is not None:
            searchers.append(rules._IniBasedRulesSearcher(text2.splitlines()))
        return rules._StackedRulesSearcher(searchers)

    def test_stack_searching(self):
        rs = self.make_searcher('[name ./a.txt]\nfoo=baz\n', '[name *.txt]\nfoo=bar\na=True\n')
        self.assertEqual((('foo', 'baz'),), rs.get_items('a.txt'))
        self.assertEqual('baz', rs.get_single_value('a.txt', 'foo'))
        self.assertEqual(None, rs.get_single_value('a.txt', 'a'))
        self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('dir/a.txt'))
        self.assertEqual('bar', rs.get_single_value('dir/a.txt', 'foo'))
        self.assertEqual('True', rs.get_single_value('dir/a.txt', 'a'))