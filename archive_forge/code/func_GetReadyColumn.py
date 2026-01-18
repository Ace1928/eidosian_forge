from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import kubernetes_consts
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import resource_transform
def GetReadyColumn():
    return 'aliases.%s.enum(status).color(%s="%s",%s="%s",%s="%s"):alias=STATUS:label=""' % (READY_COLUMN_ALIAS_KEY, GetReadyColor(kubernetes_consts.VAL_FALSE), GetReadySymbol(kubernetes_consts.VAL_FALSE), GetReadyColor(kubernetes_consts.VAL_TRUE), GetReadySymbol(kubernetes_consts.VAL_TRUE), GetReadyColor(kubernetes_consts.VAL_UNKNOWN), GetReadySymbol(kubernetes_consts.VAL_UNKNOWN))