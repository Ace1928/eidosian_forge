from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
from apitools.base.py import encoding
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.compute import ssh_troubleshooter
from googlecloudsdk.command_lib.compute import ssh_troubleshooter_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console.console_io import OperationCancelledError
def _CheckVMStatus(self):
    if self.instance.status != self.compute_message.Instance.StatusValueValuesEnum.RUNNING:
        self.issues['vm_status'] = VM_STATUS_MESSAGE