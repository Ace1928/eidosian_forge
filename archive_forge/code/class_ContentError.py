from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import files
class ContentError(Error):
    """Error if content is not given."""