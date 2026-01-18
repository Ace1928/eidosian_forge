from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def _GetExampleResource(client, track):
    """Gets an example URL Map."""
    backend_service_uri_prefix = 'https://compute.googleapis.com/compute/%(track)s/projects/my-project/global/backendServices/' % {'track': track}
    backend_bucket_uri_prefix = 'https://compute.googleapis.com/compute/%(track)s/projects/my-project/global/backendBuckets/' % {'track': track}
    return client.messages.UrlMap(name='site-map', defaultService=backend_service_uri_prefix + 'default-service', hostRules=[client.messages.HostRule(hosts=['*.google.com', 'google.com'], pathMatcher='www'), client.messages.HostRule(hosts=['*.youtube.com', 'youtube.com', '*-youtube.com'], pathMatcher='youtube')], pathMatchers=[client.messages.PathMatcher(name='www', defaultService=backend_service_uri_prefix + 'www-default', pathRules=[client.messages.PathRule(paths=['/search', '/search/*'], service=backend_service_uri_prefix + 'search'), client.messages.PathRule(paths=['/search/ads', '/search/ads/*'], service=backend_service_uri_prefix + 'ads'), client.messages.PathRule(paths=['/images/*'], service=backend_bucket_uri_prefix + 'images')]), client.messages.PathMatcher(name='youtube', defaultService=backend_service_uri_prefix + 'youtube-default', pathRules=[client.messages.PathRule(paths=['/search', '/search/*'], service=backend_service_uri_prefix + 'youtube-search'), client.messages.PathRule(paths=['/watch', '/view', '/preview'], service=backend_service_uri_prefix + 'youtube-watch')])], tests=[client.messages.UrlMapTest(host='www.google.com', path='/search/ads/inline?q=flowers', service=backend_service_uri_prefix + 'ads'), client.messages.UrlMapTest(host='youtube.com', path='/watch/this', service=backend_service_uri_prefix + 'youtube-default'), client.messages.UrlMapTest(host='youtube.com', path='/images/logo.png', service=backend_bucket_uri_prefix + 'images')])