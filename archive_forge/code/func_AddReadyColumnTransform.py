from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_transform
def AddReadyColumnTransform(parser):
    """Adds the transformation to correctly display the 'Ready'column.

  The transformation converts the status values of True/False/Unknown into
  corresponding symbols.

  Args:
    parser: parser object to add the transformation to.
  """
    status = {kubernetes_consts.VAL_TRUE: GetReadySymbol(kubernetes_consts.VAL_TRUE), kubernetes_consts.VAL_FALSE: GetReadySymbol(kubernetes_consts.VAL_FALSE), kubernetes_consts.VAL_UNKNOWN: GetReadySymbol(kubernetes_consts.VAL_UNKNOWN)}
    transforms = {resource_transform.GetTypeDataName('status', 'enum'): status}
    parser.display_info.AddTransforms(transforms)