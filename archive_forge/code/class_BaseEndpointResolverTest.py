import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
class BaseEndpointResolverTest(unittest.TestCase):

    def _endpoint_data(self):
        return {'partitions': [{'partition': 'aws', 'dnsSuffix': 'amazonaws.com', 'regionRegex': '^(us|eu)\\-\\w+$', 'defaults': {'hostname': '{service}.{region}.{dnsSuffix}'}, 'regions': {'us-foo': {'regionName': 'a'}, 'us-bar': {'regionName': 'b'}, 'eu-baz': {'regionName': 'd'}}, 'services': {'ec2': {'endpoints': {'us-foo': {}, 'us-bar': {}, 'eu-baz': {}, 'd': {}}}, 's3': {'defaults': {'sslCommonName': '{service}.{region}.{dnsSuffix}'}, 'endpoints': {'us-foo': {'sslCommonName': '{region}.{service}.{dnsSuffix}'}, 'us-bar': {}, 'eu-baz': {'hostname': 'foo'}}}, 'not-regionalized': {'isRegionalized': False, 'partitionEndpoint': 'aws', 'endpoints': {'aws': {'hostname': 'not-regionalized'}, 'us-foo': {}, 'eu-baz': {}}}, 'non-partition': {'partitionEndpoint': 'aws', 'endpoints': {'aws': {'hostname': 'host'}, 'us-foo': {}}}, 'merge': {'defaults': {'signatureVersions': ['v2'], 'protocols': ['http']}, 'endpoints': {'us-foo': {'signatureVersions': ['v4']}, 'us-bar': {'protocols': ['https']}}}}}, {'partition': 'foo', 'dnsSuffix': 'foo.com', 'regionRegex': '^(foo)\\-\\w+$', 'defaults': {'hostname': '{service}.{region}.{dnsSuffix}', 'protocols': ['http'], 'foo': 'bar'}, 'regions': {'foo-1': {'regionName': '1'}, 'foo-2': {'regionName': '2'}, 'foo-3': {'regionName': '3'}}, 'services': {'ec2': {'endpoints': {'foo-1': {'foo': 'baz'}, 'foo-2': {}, 'foo-3': {}}}}}]}