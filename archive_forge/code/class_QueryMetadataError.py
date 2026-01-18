from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions as core_exceptions
class QueryMetadataError(Error):
    """Class for errors raised when querying metadata."""