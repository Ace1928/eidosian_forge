from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_transform
def GetReadyColor(ready):
    if ready == kubernetes_consts.VAL_UNKNOWN:
        return 'yellow'
    elif ready == kubernetes_consts.VAL_TRUE or ready == kubernetes_consts.VAL_READY:
        return 'green'
    else:
        return 'red'