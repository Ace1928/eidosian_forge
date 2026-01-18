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
def AddAirflowConfigUpdateFlagsToGroup(update_type_group):
    """Adds flags related to updating Airflow configurations.

  Args:
    update_type_group: argument group, the group to which flags should be added.
  """
    _AddPartialDictUpdateFlagsToGroup(update_type_group, CLEAR_AIRFLOW_CONFIGS_FLAG, REMOVE_AIRFLOW_CONFIGS_FLAG, UPDATE_AIRFLOW_CONFIGS_FLAG, AIRFLOW_CONFIGS_FLAG_GROUP_DESCRIPTION)