from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from typing import Mapping, Sequence
from googlecloudsdk.api_lib.run import container_resource
from googlecloudsdk.command_lib.run.printers import k8s_object_printer_util as k8s_util
from googlecloudsdk.core.resource import custom_printer_base as cp
def _FormatVolumeMount(name, volume):
    """Format details about a volume mount."""
    if not volume:
        return 'volume not found'
    if volume.emptyDir:
        return cp.Labeled([('name', name), ('type', 'in-memory')])
    elif volume.nfs:
        return cp.Labeled([('name', name), ('type', 'nfs'), ('location', '{}:{}'.format(volume.nfs.server, volume.nfs.path))])
    elif volume.csi:
        if volume.csi.driver == 'gcsfuse.run.googleapis.com':
            bucket = None
            for prop in volume.csi.volumeAttributes.additionalProperties:
                if prop.key == 'bucketName':
                    bucket = prop.value
            return cp.Labeled([('name', name), ('type', 'cloud-storage'), ('bucket', bucket)])