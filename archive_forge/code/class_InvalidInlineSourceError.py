from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.core import exceptions
import six
class InvalidInlineSourceError(exceptions.Error):
    """Error if a inline source is improperly specified."""