from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def _ValidateContact(flag_value):
    if not re.match('^[0-9]+$', flag_value):
        raise exceptions.InvalidArgumentException('contact', flag_value)