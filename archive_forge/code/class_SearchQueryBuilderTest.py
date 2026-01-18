from oslotest import base
from aodhclient import utils
class SearchQueryBuilderTest(base.BaseTestCase):

    def _do_test(self, expr, expected):
        req = utils.search_query_builder(expr)
        self.assertEqual(expected, req)

    def test_search_query_builder(self):
        self._do_test('foo=bar', {'=': {'foo': 'bar'}})
        self._do_test('foo!=1', {'!=': {'foo': 1.0}})
        self._do_test('foo=True', {'=': {'foo': True}})
        self._do_test('foo=null', {'=': {'foo': None}})
        self._do_test('foo="null"', {'=': {'foo': 'null'}})
        self._do_test('not (foo="quote" or foo="what!" or bar="who?")', {'not': {'or': [{'=': {'bar': 'who?'}}, {'=': {'foo': 'what!'}}, {'=': {'foo': 'quote'}}]}})
        self._do_test('(foo="quote" or not foo="what!" or bar="who?") and cat="meme"', {'and': [{'=': {'cat': 'meme'}}, {'or': [{'=': {'bar': 'who?'}}, {'not': {'=': {'foo': 'what!'}}}, {'=': {'foo': 'quote'}}]}]})
        self._do_test('foo="quote" or foo="what!" or bar="who?" and cat="meme"', {'or': [{'and': [{'=': {'cat': 'meme'}}, {'=': {'bar': 'who?'}}]}, {'=': {'foo': 'what!'}}, {'=': {'foo': 'quote'}}]})
        self._do_test('foo="quote" and foo="what!" or bar="who?" or cat="meme"', {'or': [{'=': {'cat': 'meme'}}, {'=': {'bar': 'who?'}}, {'and': [{'=': {'foo': 'what!'}}, {'=': {'foo': 'quote'}}]}]})