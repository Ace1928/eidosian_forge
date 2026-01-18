from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.kuberun import structuredout
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kuberun import flags
from googlecloudsdk.command_lib.kuberun import k8s_object_printer
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.command_lib.kuberun import kuberun_command
from googlecloudsdk.command_lib.kuberun import pretty_print
from googlecloudsdk.core import exceptions
def _AddAliases(mapping):
    """Add aliases to embedded fields displayed in the output.

  Adds aliases to embedded fields that would require a more complex expression
  to be shown in the output table.

  Args:
   mapping: a domain mapping unmarshalled from json

  Returns:
   dictionary with aliases representing the domain mapping from the input
  """
    d = structuredout.DictWithAliases(**mapping)
    ready_cond = k8s_object_printer.ReadyConditionFromDict(mapping)
    if ready_cond is not None:
        d.AddAlias(pretty_print.READY_COLUMN_ALIAS_KEY, ready_cond.get(kubernetes_consts.FIELD_STATUS, kubernetes_consts.VAL_UNKNOWN))
    return d