from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import service
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.command_lib.run.printers import revision_printer
from googlecloudsdk.command_lib.run.printers import traffic_printer
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
def _GetWorkerSettings(self, record):
    """Adds worker-level values."""
    labels = [cp.Labeled([('Binary Authorization', k8s_util.GetBinAuthzPolicy(record)), ('Min Instances', GetMinInstances(record))])]
    breakglass_value = k8s_util.GetBinAuthzBreakglass(record)
    if breakglass_value is not None:
        breakglass_label = cp.Labeled([('Breakglass Justification', breakglass_value)])
        breakglass_label.skip_empty = False
        labels.append(breakglass_label)
    description = k8s_util.GetDescription(record)
    if description is not None:
        description_label = cp.Labeled([('Description', description)])
        labels.append(description_label)
    return cp.Section(labels)