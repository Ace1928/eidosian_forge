from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import ipaddress
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import properties
import six
def AddPypiUpdateFlagsToGroup(update_type_group):
    """Adds flag related to setting Pypi updates.

  Args:
    update_type_group: argument group, the group to which flag should be added.
  """
    group = update_type_group.add_mutually_exclusive_group(PYPI_PACKAGES_FLAG_GROUP_DESCRIPTION)
    UPDATE_PYPI_FROM_FILE_FLAG.AddToParser(group)
    _AddPartialDictUpdateFlagsToGroup(group, CLEAR_PYPI_PACKAGES_FLAG, REMOVE_PYPI_PACKAGES_FLAG, UPDATE_PYPI_PACKAGE_FLAG)