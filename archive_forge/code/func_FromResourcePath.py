from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.app import env
from googlecloudsdk.api_lib.app import metric_names
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import metrics
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import text
from googlecloudsdk.core.util import times
import six
from six.moves import map  # pylint: disable=redefined-builtin
@classmethod
def FromResourcePath(cls, path):
    parts = path.split('/')
    if not 0 < len(parts) <= cls._RESOURCE_PATH_PARTS:
        raise VersionValidationError('[{0}] is not a valid resource path. Expected <project>/<service>/<version>')
    parts = [None] * (cls._RESOURCE_PATH_PARTS - len(parts)) + parts
    return cls(*parts)