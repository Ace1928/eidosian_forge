from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
class ArgMultiValueDict:
    """Converts argument values into multi-valued mappings.

  Values for the repeated keys are collected in a list.
  """

    def __init__(self):
        ops = '='
        key_op_value_pattern = '([^{ops}]+)([{ops}]?)(.*)'.format(ops=ops)
        self._key_op_value = re.compile(key_op_value_pattern, re.DOTALL)

    def __call__(self, arg_value):
        arg_list = [item.strip() for item in arg_value.split(',')]
        arg_dict = collections.OrderedDict()
        for arg in arg_list:
            match = self._key_op_value.match(arg)
            if not match:
                raise arg_parsers.ArgumentTypeError('Invalid flag value [{0}]'.format(arg))
            key, _, value = (match.group(1).strip(), match.group(2), match.group(3).strip())
            arg_dict.setdefault(key, []).append(value)
        return arg_dict