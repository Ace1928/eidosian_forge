from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _WaitMinToSec(wait):
    if not wait:
        return wait
    if not re.fullmatch('\\d+m', wait):
        raise exceptions.AutomationWaitFormatError()
    mins = wait[:-1]
    seconds = int(mins) * 60
    return '%ss' % seconds