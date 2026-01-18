import copy
import json
import os
from googlecloudsdk.command_lib.anthos.common import messages
from googlecloudsdk.command_lib.util.anthos import binary_operations
from googlecloudsdk.core import exceptions as c_except
from googlecloudsdk.core.credentials import store as c_store
def _ParseBugReportArgs(self, context, **kwargs):
    del kwargs
    exec_args = ['bug-report', '--context', context]
    return exec_args