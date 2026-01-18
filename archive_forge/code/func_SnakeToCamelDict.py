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
def SnakeToCamelDict(arg_type):
    """Reccursive method to convert all nested snake case dictionary keys to camel case."""
    if isinstance(arg_type, list):
        return [SnakeToCamelDict(list_val) if isinstance(list_val, (dict, list)) else list_val for list_val in arg_type]
    return {SnakeToCamel(key): SnakeToCamelDict(value) if isinstance(value, (dict, list)) else value for key, value in arg_type.items()}