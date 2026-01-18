from tests.compat import mock, unittest
from httpretty import HTTPretty
import json
import requests
from boto.cloudsearch.search import SearchConnection, SearchServiceException
from boto.compat import six, map
class CloudSearchSearchTest(CloudSearchSearchBaseTest):
    response = {'rank': '-text_relevance', 'match-expr': 'Test', 'hits': {'found': 30, 'start': 0, 'hit': CloudSearchSearchBaseTest.hits}, 'info': {'rid': 'b7c167f6c2da6d93531b9a7b314ad030b3a74803b4b7797edb905ba5a6a08', 'time-ms': 2, 'cpu-time-ms': 0}}

    def test_cloudsearch_qsearch(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test')
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'q'], [b'Test'])
        self.assertEqual(args[b'start'], [b'0'])
        self.assertEqual(args[b'size'], [b'10'])

    def test_cloudsearch_bqsearch(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(bq="'Test'")
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'bq'], [b"'Test'"])

    def test_cloudsearch_search_details(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', size=50, start=20)
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'q'], [b'Test'])
        self.assertEqual(args[b'size'], [b'50'])
        self.assertEqual(args[b'start'], [b'20'])

    def test_cloudsearch_facet_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet=['Author'])
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet'], [b'Author'])

    def test_cloudsearch_facet_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet=['author', 'cat'])
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet'], [b'author,cat'])

    def test_cloudsearch_facet_constraint_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet_constraints={'author': "'John Smith','Mark Smith'"})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet-author-constraints'], [b"'John Smith','Mark Smith'"])

    def test_cloudsearch_facet_constraint_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet_constraints={'author': "'John Smith','Mark Smith'", 'category': "'News','Reviews'"})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet-author-constraints'], [b"'John Smith','Mark Smith'"])
        self.assertEqual(args[b'facet-category-constraints'], [b"'News','Reviews'"])

    def test_cloudsearch_facet_sort_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet_sort={'author': 'alpha'})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet-author-sort'], [b'alpha'])

    def test_cloudsearch_facet_sort_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet_sort={'author': 'alpha', 'cat': 'count'})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet-author-sort'], [b'alpha'])
        self.assertEqual(args[b'facet-cat-sort'], [b'count'])

    def test_cloudsearch_top_n_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet_top_n={'author': 5})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet-author-top-n'], [b'5'])

    def test_cloudsearch_top_n_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', facet_top_n={'author': 5, 'cat': 10})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'facet-author-top-n'], [b'5'])
        self.assertEqual(args[b'facet-cat-top-n'], [b'10'])

    def test_cloudsearch_rank_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', rank=['date'])
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'rank'], [b'date'])

    def test_cloudsearch_rank_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', rank=['date', 'score'])
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'rank'], [b'date,score'])

    def test_cloudsearch_result_fields_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', return_fields=['author'])
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'return-fields'], [b'author'])

    def test_cloudsearch_result_fields_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', return_fields=['author', 'title'])
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b'return-fields'], [b'author,title'])

    def test_cloudsearch_t_field_single(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', t={'year': '2001..2007'})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b't-year'], [b'2001..2007'])

    def test_cloudsearch_t_field_multiple(self):
        search = SearchConnection(endpoint=HOSTNAME)
        search.search(q='Test', t={'year': '2001..2007', 'score': '10..50'})
        args = self.get_args(HTTPretty.last_request.raw_requestline)
        self.assertEqual(args[b't-year'], [b'2001..2007'])
        self.assertEqual(args[b't-score'], [b'10..50'])

    def test_cloudsearch_results_meta(self):
        """Check returned metadata is parsed correctly"""
        search = SearchConnection(endpoint=HOSTNAME)
        results = search.search(q='Test')
        self.assertEqual(results.rank, '-text_relevance')
        self.assertEqual(results.match_expression, 'Test')

    def test_cloudsearch_results_info(self):
        """Check num_pages_needed is calculated correctly"""
        search = SearchConnection(endpoint=HOSTNAME)
        results = search.search(q='Test')
        self.assertEqual(results.num_pages_needed, 3.0)

    def test_cloudsearch_results_matched(self):
        """
        Check that information objects are passed back through the API
        correctly.
        """
        search = SearchConnection(endpoint=HOSTNAME)
        query = search.build_query(q='Test')
        results = search(query)
        self.assertEqual(results.search_service, search)
        self.assertEqual(results.query, query)

    def test_cloudsearch_results_hits(self):
        """Check that documents are parsed properly from AWS"""
        search = SearchConnection(endpoint=HOSTNAME)
        results = search.search(q='Test')
        hits = list(map(lambda x: x['id'], results.docs))
        self.assertEqual(hits, ['12341', '12342', '12343', '12344', '12345', '12346', '12347'])

    def test_cloudsearch_results_iterator(self):
        """Check the results iterator"""
        search = SearchConnection(endpoint=HOSTNAME)
        results = search.search(q='Test')
        results_correct = iter(['12341', '12342', '12343', '12344', '12345', '12346', '12347'])
        for x in results:
            self.assertEqual(x['id'], next(results_correct))

    def test_cloudsearch_results_internal_consistancy(self):
        """Check the documents length matches the iterator details"""
        search = SearchConnection(endpoint=HOSTNAME)
        results = search.search(q='Test')
        self.assertEqual(len(results), len(results.docs))

    def test_cloudsearch_search_nextpage(self):
        """Check next page query is correct"""
        search = SearchConnection(endpoint=HOSTNAME)
        query1 = search.build_query(q='Test')
        query2 = search.build_query(q='Test')
        results = search(query2)
        self.assertEqual(results.next_page().query.start, query1.start + query1.size)
        self.assertEqual(query1.q, query2.q)