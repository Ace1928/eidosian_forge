import sys
import time
import datetime
import itertools
from libcloud.pricing import get_pricing
from libcloud.common.base import LazyObject
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.compute.providers import Provider
def _build_disk_gce_struct(self, device_name, source=None, disk_type=None, disk_size=None, image=None, disk_name=None, is_boot=True, mount_mode='READ_WRITE', usage_type='PERSISTENT', auto_delete=True, use_selflinks=True):
    """
        Generates the GCP dict for a disk.

        :param    device_name: Specifies a unique device name of your
                               choice that is reflected into the
                               /dev/disk/by-id/google-* tree
                               of a Linux operating system running within the
                               instance. This name can be used to reference the
                               device for mounting, resizing, and so on, from
                               within the instance.  Defaults to disk_name.
        :type      device_name: ``str``

        :keyword   source: The disk to attach to the instance.
        :type      source: ``str`` of selfLink, :class:`StorageVolume` or None

        :keyword   disk_type: Specify a URL or DiskType object.
        :type      disk_type: ``str`` or :class:`GCEDiskType` or ``None``

        :keyword   image: The image to use to create the disk.
        :type      image: :class:`GCENodeImage` or ``None``

        :keyword   disk_size: Integer in gigabytes.
        :type      disk_size: ``int``

        :param     disk_name: Specifies the disk name. If not specified, the
                              default is to use the device_name.
        :type      disk_name: ``str``

        :keyword   mount_mode: The mode in which to attach this disk, either
                               READ_WRITE or READ_ONLY. If not specified,
                               the default is to attach the disk in READ_WRITE
                               mode.
        :type      mount_mode: ``str``

        :keyword   usage_type: Specifies the type of the disk, either SCRATCH
                               or PERSISTENT. If not specified, the default
                               is PERSISTENT.
        :type      usage_type: ``str``

        :keyword   auto_delete: Indicate that the boot disk should be
                                deleted when the Node is deleted. Set to
                                True by default.
        :type      auto_delete: ``bool``

        :return:   Dictionary to be used in disk-portion of
                   instance API call.
        :rtype:    ``dict``
        """
    if source is None and image is None:
        raise ValueError("Either the 'source' or 'image' argument must be specified.")
    if not isinstance(auto_delete, bool):
        raise ValueError('auto_delete field is not a bool.')
    if disk_size is not None and (not (isinstance(disk_size, int) or disk_size.isdigit())):
        raise ValueError("disk_size must be a digit, '%s' provided." % str(disk_size))
    mount_modes = ['READ_WRITE', 'READ_ONLY']
    if mount_mode not in mount_modes:
        raise ValueError('mount mode must be one of: %s.' % ','.join(mount_modes))
    usage_types = ['PERSISTENT', 'SCRATCH']
    if usage_type not in usage_types:
        raise ValueError('usage type must be one of: %s.' % ','.join(usage_types))
    disk = {}
    if not disk_name:
        disk_name = device_name
    if source is not None:
        disk['source'] = self._get_selflink_or_name(obj=source, get_selflinks=use_selflinks, objname='volume')
    else:
        image = self._get_selflink_or_name(obj=image, get_selflinks=True, objname='image')
        disk_type = self._get_selflink_or_name(obj=disk_type, get_selflinks=use_selflinks, objname='disktype')
        disk['initializeParams'] = {'diskName': disk_name, 'diskType': disk_type, 'sourceImage': image}
        if disk_size is not None:
            disk['initializeParams']['diskSizeGb'] = disk_size
    disk.update({'boot': is_boot, 'type': usage_type, 'mode': mount_mode, 'deviceName': device_name, 'autoDelete': auto_delete})
    return disk