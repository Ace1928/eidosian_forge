from unittest import mock
import testtools
from heatclient.common import utils
from heatclient.v1 import resource_types
def _base_test(self, expect, key):

    class FakeAPI(object):
        """Fake API and ensure request url is correct."""

        def get(self, *args, **kwargs):
            assert ('GET', args[0]) == expect

        def json_request(self, *args, **kwargs):
            assert args == expect
            ret = key and {key: []} or {}
            return ({}, {key: ret})

        def raw_request(self, *args, **kwargs):
            assert args == expect
            return {}

        def head(self, url, **kwargs):
            return self.json_request('HEAD', url, **kwargs)

        def post(self, url, **kwargs):
            return self.json_request('POST', url, **kwargs)

        def put(self, url, **kwargs):
            return self.json_request('PUT', url, **kwargs)

        def delete(self, url, **kwargs):
            return self.raw_request('DELETE', url, **kwargs)

        def patch(self, url, **kwargs):
            return self.json_request('PATCH', url, **kwargs)
    manager = resource_types.ResourceTypeManager(FakeAPI())
    return manager