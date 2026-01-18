from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
class CacheKeyQueryStringException(core_exceptions.Error):

    def __init__(self):
        super(CacheKeyQueryStringException, self).__init__('cache-key-query-string-whitelist and cache-key-query-string-blacklist may only be set when cache-key-include-query-string is enabled.')