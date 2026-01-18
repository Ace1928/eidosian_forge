import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestResourceFind(base.TestCase):
    result = 1

    class Base(resource.Resource):

        @classmethod
        def existing(cls, **kwargs):
            response = mock.Mock()
            response.status_code = 404
            raise exceptions.ResourceNotFound('Not Found', response=response)

        @classmethod
        def list(cls, session, **params):
            return []

    class OneResult(Base):

        @classmethod
        def _get_one_match(cls, *args):
            return TestResourceFind.result

    class NoResults(Base):

        @classmethod
        def _get_one_match(cls, *args):
            return None

    class OneResultWithQueryParams(OneResult):
        _query_mapping = resource.QueryParameters('name')

    def setUp(self):
        super(TestResourceFind, self).setUp()
        self.no_results = self.NoResults
        self.one_result = self.OneResult
        self.one_result_with_qparams = self.OneResultWithQueryParams

    def test_find_short_circuit(self):
        value = 1

        class Test(resource.Resource):

            @classmethod
            def existing(cls, **kwargs):
                mock_match = mock.Mock()
                mock_match.fetch.return_value = value
                return mock_match
        result = Test.find(self.cloud.compute, 'name')
        self.assertEqual(result, value)

    def test_no_match_raise(self):
        self.assertRaises(exceptions.ResourceNotFound, self.no_results.find, self.cloud.compute, 'name', ignore_missing=False)

    def test_no_match_return(self):
        self.assertIsNone(self.no_results.find(self.cloud.compute, 'name', ignore_missing=True))

    def test_find_result_name_not_in_query_parameters(self):
        with mock.patch.object(self.one_result, 'existing', side_effect=self.OneResult.existing) as mock_existing, mock.patch.object(self.one_result, 'list', side_effect=self.OneResult.list) as mock_list:
            self.assertEqual(self.result, self.one_result.find(self.cloud.compute, 'name'))
            mock_existing.assert_called_once_with(id='name', connection=mock.ANY)
            mock_list.assert_called_once_with(mock.ANY)

    def test_find_result_name_in_query_parameters(self):
        self.assertEqual(self.result, self.one_result_with_qparams.find(self.cloud.compute, 'name'))

    def test_match_empty_results(self):
        self.assertIsNone(resource.Resource._get_one_match('name', []))

    def test_no_match_by_name(self):
        the_name = 'Brian'
        match = mock.Mock(spec=resource.Resource)
        match.name = the_name
        result = resource.Resource._get_one_match('Richard', [match])
        self.assertIsNone(result, match)

    def test_single_match_by_name(self):
        the_name = 'Brian'
        match = mock.Mock(spec=resource.Resource)
        match.name = the_name
        result = resource.Resource._get_one_match(the_name, [match])
        self.assertIs(result, match)

    def test_single_match_by_id(self):
        the_id = 'Brian'
        match = mock.Mock(spec=resource.Resource)
        match.id = the_id
        result = resource.Resource._get_one_match(the_id, [match])
        self.assertIs(result, match)

    def test_single_match_by_alternate_id(self):
        the_id = 'Richard'

        class Test(resource.Resource):
            other_id = resource.Body('other_id', alternate_id=True)
        match = Test(other_id=the_id)
        result = Test._get_one_match(the_id, [match])
        self.assertIs(result, match)

    def test_multiple_matches(self):
        the_id = 'Brian'
        match = mock.Mock(spec=resource.Resource)
        match.id = the_id
        self.assertRaises(exceptions.DuplicateResource, resource.Resource._get_one_match, the_id, [match, match])

    def test_list_no_base_path(self):
        with mock.patch.object(self.Base, 'list') as list_mock:
            self.Base.find(self.cloud.compute, 'name')
            list_mock.assert_called_with(self.cloud.compute)

    def test_list_base_path(self):
        with mock.patch.object(self.Base, 'list') as list_mock:
            self.Base.find(self.cloud.compute, 'name', list_base_path='/dummy/list')
            list_mock.assert_called_with(self.cloud.compute, base_path='/dummy/list')