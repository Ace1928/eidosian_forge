from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def _HandleCloudStorageConfigUpdate(self, update_setting):
    if update_setting.value == CLEAR_CLOUD_STORAGE_CONFIG_VALUE:
        update_setting.value = None