from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import io
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
def FetchExecutionSpecArgs(args_map_as_list):
    """Returns Dataplex task execution spec args as a map of key,value pairs from an input list of strings of format key=value."""
    execution_args_map = dict()
    for arg_entry in args_map_as_list:
        if '=' not in arg_entry:
            raise argparse.ArgumentTypeError("Execution spec argument '{}' should be of the type argKey=argValue.".format(arg_entry))
        arg_entry_split = arg_entry.split('=', 1)
        if len(arg_entry_split) < 2 or len(arg_entry_split[0].strip()) == 0 or len(arg_entry_split[1]) == 0:
            raise argparse.ArgumentTypeError("Execution spec argument '{}' should be of the format argKey=argValue.".format(arg_entry))
        execution_args_map[arg_entry_split[0]] = arg_entry_split[1]
    return execution_args_map