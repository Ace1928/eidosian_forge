from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.kuberun import revision
from googlecloudsdk.command_lib.kuberun import k8s_object_printer
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core.resource import custom_printer_base as cp
import six
def Active(record):
    """Returns True/False/None indicating the active status of the resource."""
    active_cond = [x for x in record.get(kubernetes_consts.FIELD_STATUS, {}).get(kubernetes_consts.FIELD_CONDITIONS, []) if x[kubernetes_consts.FIELD_TYPE] == kubernetes_consts.VAL_ACTIVE]
    if active_cond:
        status = active_cond[0].get(kubernetes_consts.FIELD_STATUS)
        return True if status == kubernetes_consts.VAL_TRUE else False
    else:
        return None