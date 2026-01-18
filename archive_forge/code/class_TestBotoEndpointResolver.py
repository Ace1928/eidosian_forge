import mock
import os
import json
from nose.tools import assert_equal
from tests.unit import unittest
import boto
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
class TestBotoEndpointResolver(BaseEndpointResolverTest):

    def test_get_all_available_regions(self):
        resolver = BotoEndpointResolver(self._endpoint_data())
        regions = sorted(resolver.get_all_available_regions('ec2'))
        expected_regions = sorted(['us-bar', 'eu-baz', 'us-foo', 'foo-1', 'foo-2', 'foo-3'])
        self.assertEqual(regions, expected_regions)

    def test_get_all_regions_on_non_regional_service(self):
        resolver = BotoEndpointResolver(self._endpoint_data())
        regions = sorted(resolver.get_all_available_regions('not-regionalized'))
        expected_regions = sorted(['us-foo', 'us-bar', 'eu-baz'])
        self.assertEqual(regions, expected_regions)

    def test_get_all_regions_with_renames(self):
        rename_map = {'ec3': 'ec2'}
        resolver = BotoEndpointResolver(endpoint_data=self._endpoint_data(), service_rename_map=rename_map)
        regions = sorted(resolver.get_all_available_regions('ec2'))
        expected_regions = sorted(['us-bar', 'eu-baz', 'us-foo', 'foo-1', 'foo-2', 'foo-3'])
        self.assertEqual(regions, expected_regions)

    def test_resolve_hostname(self):
        resolver = BotoEndpointResolver(self._endpoint_data())
        hostname = resolver.resolve_hostname('ec2', 'us-foo')
        expected_hostname = 'ec2.us-foo.amazonaws.com'
        self.assertEqual(hostname, expected_hostname)

    def test_resolve_hostname_with_rename(self):
        rename_map = {'ec3': 'ec2'}
        resolver = BotoEndpointResolver(endpoint_data=self._endpoint_data(), service_rename_map=rename_map)
        hostname = resolver.resolve_hostname('ec3', 'us-foo')
        expected_hostname = 'ec2.us-foo.amazonaws.com'
        self.assertEqual(hostname, expected_hostname)

    def test_resolve_hostname_with_ssl_common_name(self):
        resolver = BotoEndpointResolver(self._endpoint_data())
        hostname = resolver.resolve_hostname('s3', 'us-foo')
        expected_hostname = 'us-foo.s3.amazonaws.com'
        self.assertEqual(hostname, expected_hostname)

    def test_resolve_hostname_on_invalid_region_prefix(self):
        resolver = BotoEndpointResolver(self._endpoint_data())
        hostname = resolver.resolve_hostname('s3', 'fake-west-1')
        self.assertIsNone(hostname)

    def test_get_available_services(self):
        resolver = BotoEndpointResolver(self._endpoint_data())
        services = sorted(resolver.get_available_services())
        expected_services = sorted(['ec2', 's3', 'not-regionalized', 'non-partition', 'merge'])
        self.assertEqual(services, expected_services)

    def test_get_available_services_with_renames(self):
        rename_map = {'ec3': 'ec2'}
        resolver = BotoEndpointResolver(endpoint_data=self._endpoint_data(), service_rename_map=rename_map)
        services = sorted(resolver.get_available_services())
        expected_services = sorted(['ec3', 's3', 'not-regionalized', 'non-partition', 'merge'])
        self.assertEqual(services, expected_services)