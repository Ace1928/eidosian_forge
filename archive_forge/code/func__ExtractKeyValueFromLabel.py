from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
from googlecloudsdk.api_lib.ml.vision import api_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
def _ExtractKeyValueFromLabel(label):
    """Extracts key and value from label like 'key=value'.

  Args:
    label: str, the label to extract key and values, i.e. 'foo=buz'.

  Returns:
    (k, v): k is the substring before '=', v is the substring after '='.

  Raises:
    LabelsFormatError, raises when label is not formatted as 'key=value', or
    key or value is empty.
  """
    try:
        k, v = label.split('=')
        if k and v:
            return (k, v)
        raise ValueError('Key or value cannot be empty string.')
    except ValueError:
        raise LabelsFormatError('Each label must be formatted as "key=value". key and value cannot be empty.')