from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _HandlePushNoWrapperUpdate(self, update_setting):
    if update_setting.value == CLEAR_PUSH_NO_WRAPPER_CONFIG_VALUE:
        update_setting.value = None