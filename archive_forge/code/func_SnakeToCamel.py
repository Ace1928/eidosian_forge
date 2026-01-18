from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import re
from apitools.base.protorpclite import messages as proto_messages
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import files
import six
def SnakeToCamel(msg, skip=None):
    """Recursively transform all keys and values from snake_case to camelCase.

  If a key is in skip, then its value is left alone.

  Args:
    msg: dict, list, or other. If 'other', the function returns immediately.
    skip: contains dict keys whose values should not have camel case applied.

  Returns:
    Same type as msg, except all strings that were snake_case are now CamelCase,
    except for the values of dict keys contained in skip.
  """
    if skip is None:
        skip = []
    if isinstance(msg, dict):
        return {SnakeToCamelString(key): SnakeToCamel(val, skip) if key not in skip else val for key, val in six.iteritems(msg)}
    elif isinstance(msg, list):
        return [SnakeToCamel(elem, skip) for elem in msg]
    else:
        return msg