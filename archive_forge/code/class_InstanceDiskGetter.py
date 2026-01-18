from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
class InstanceDiskGetter(object):
    """Class returning disks of existing instance, with lazy loading."""

    def __init__(self, instance_ref, holder):
        self._instance_ref = instance_ref
        self._holder = holder
        self._instance_disks = None
        self.instance_exists = None

    def _get_instance_disks_maybe(self):
        if self._instance_disks is None:
            request = self._holder.client.messages.ComputeInstancesGetRequest(instance=self._instance_ref.Name(), project=self._instance_ref.project, zone=self._instance_ref.zone)
            try:
                instance = self._holder.client.apitools_client.instances.Get(request)
                self.instance_exists = True
                self._instance_disks = instance.disks
            except apitools_exceptions.HttpNotFoundError:
                self.instance_exists = False
                self._instance_disks = []

    def get_disk(self, device_name):
        """Return instance's disk with given name."""
        self._get_instance_disks_maybe()
        for disk in self._instance_disks:
            if disk.deviceName == device_name:
                return disk