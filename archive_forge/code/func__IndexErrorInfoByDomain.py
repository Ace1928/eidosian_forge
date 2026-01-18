from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _IndexErrorInfoByDomain(self, details):
    """Extracts and indexes error info list by the domain attribute."""
    domain_map = collections.defaultdict(list)
    for item in details:
        error_type = item.get('@type', None)
        if error_type.endswith(ERROR_INFO_SUFFIX):
            domain = item.get('domain', None)
            if domain:
                domain_map[domain].append(item)
    return domain_map