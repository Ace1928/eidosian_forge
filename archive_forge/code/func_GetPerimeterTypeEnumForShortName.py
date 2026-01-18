from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.accesscontextmanager import acm_printer
from googlecloudsdk.api_lib.accesscontextmanager import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.accesscontextmanager import common
from googlecloudsdk.command_lib.accesscontextmanager import levels
from googlecloudsdk.command_lib.accesscontextmanager import policies
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def GetPerimeterTypeEnumForShortName(perimeter_type_short_name, api_version):
    """Returns the PerimeterTypeValueValuesEnum value for the given short name.

  Args:
    perimeter_type_short_name: Either 'regular' or 'bridge'.
    api_version: One of 'v1alpha', 'v1beta', or 'v1'.

  Returns:
    The appropriate value of type PerimeterTypeValueValuesEnum.
  """
    if perimeter_type_short_name is None:
        return None
    return GetTypeEnumMapper(version=api_version).GetEnumForChoice(perimeter_type_short_name)