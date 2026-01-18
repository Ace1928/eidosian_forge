from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import flags
from googlecloudsdk.command_lib.storage import name_expansion
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task_executor
from googlecloudsdk.command_lib.storage.tasks.cat import cat_task_iterator
def _range_parser(string_value):
    """Creates Range object out of given string value.

  Args:
    string_value (str): The range the user entered.

  Returns:
    Range(int, int|None): The Range object from the given string value.
  """
    if string_value == '-':
        return arg_parsers.Range(start=0, end=None)
    range_start, _, range_end = string_value.partition('-')
    if not range_start:
        return arg_parsers.Range(start=-1 * int(range_end), end=None)
    if not range_end:
        return arg_parsers.Range(start=int(range_start), end=None)
    return arg_parsers.Range.Parse(string_value)