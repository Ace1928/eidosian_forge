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
def _GetConfigMaps(self, record, container):
    config_maps = {}
    config_maps.update({k: FormatConfigMapKeyRef(v) for k, v in container.env.config_maps.items()})
    config_maps.update({k: FormatConfigMapVolumeSource(v) for k, v in record.MountedVolumeJoin(container, 'config_maps').items()})
    return cp.Mapped(k8s_object_printer.OrderByKey(config_maps))