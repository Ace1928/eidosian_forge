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
def _create_instance_properties(self, name, node_size, source=None, image=None, disk_type='pd-standard', disk_auto_delete=True, network='default', subnetwork=None, external_ip='ephemeral', internal_ip=None, can_ip_forward=None, service_accounts=None, on_host_maintenance=None, automatic_restart=None, preemptible=None, tags=None, metadata=None, description=None, disks_gce_struct=None, nic_gce_struct=None, use_selflinks=True, labels=None, accelerator_type=None, accelerator_count=None, disk_size=None):
    """
        Create the GCE instance properties needed for instance templates.

        :param    node_size: The machine type to use.
        :type     node_size: ``str`` or :class:`GCENodeSize`

        :keyword  source: A source disk to attach to the instance. Cannot
                          specify both 'image' and 'source'.
        :type     source: :class:`StorageVolume` or ``str`` or ``None``

        :param    image: The image to use to create the node. Cannot specify
                         both 'image' and 'source'.
        :type     image: ``str`` or :class:`GCENodeImage` or ``None``

        :keyword  disk_type: Specify a pd-standard (default) disk or pd-ssd
                             for an SSD disk.
        :type     disk_type: ``str`` or :class:`GCEDiskType`

        :keyword  disk_auto_delete: Indicate that the boot disk should be
                                    deleted when the Node is deleted. Set to
                                    True by default.
        :type     disk_auto_delete: ``bool``

        :keyword  network: The network to associate with the node.
        :type     network: ``str`` or :class:`GCENetwork`

        :keyword  subnetwork: The Subnetwork resource for this instance. If
                              the network resource is in legacy mode, do not
                              provide this property. If the network is in auto
                              subnet mode, providing the subnetwork is
                              optional. If the network is in custom subnet
                              mode, then this field should be specified.
        :type     subnetwork: :class: `GCESubnetwork` or None

        :keyword  external_ip: The external IP address to use.  If 'ephemeral'
                               (default), a new non-static address will be
                               used.  If 'None', then no external address will
                               be used.  To use an existing static IP address,
                               a GCEAddress object should be passed in.
        :type     external_ip: :class:`GCEAddress` or ``str`` or ``None``

        :keyword  internal_ip: The private IP address to use.
        :type     internal_ip: :class:`GCEAddress` or ``str`` or ``None``

        :keyword  can_ip_forward: Set to ``True`` to allow this node to
                                  send/receive non-matching src/dst packets.
        :type     can_ip_forward: ``bool`` or ``None``

        :keyword  service_accounts: Specify a list of serviceAccounts when
                                    creating the instance. The format is a
                                    list of dictionaries containing email
                                    and list of scopes, e.g.
                                    [{'email':'default',
                                    'scopes':['compute', ...]}, ...]
                                    Scopes can either be full URLs or short
                                    names. If not provided, use the
                                    'default' service account email and a
                                    scope of 'devstorage.read_only'. Also
                                    accepts the aliases defined in
                                    'gcloud compute'.
        :type     service_accounts: ``list``

        :keyword  on_host_maintenance: Defines whether node should be
                                       terminated or migrated when host
                                       machine goes down. Acceptable values
                                       are: 'MIGRATE' or 'TERMINATE' (If
                                       not supplied, value will be reset to
                                       GCE default value for the instance
                                       type.)
        :type     on_host_maintenance: ``str`` or ``None``

        :keyword  automatic_restart: Defines whether the instance should be
                                     automatically restarted when it is
                                     terminated by Compute Engine. (If not
                                     supplied, value will be set to the GCE
                                     default value for the instance type.)
        :type     automatic_restart: ``bool`` or ``None``

        :keyword  preemptible: Defines whether the instance is preemptible.
                               (If not supplied, the instance will not be
                               preemptible)
        :type     preemptible: ``bool`` or ``None``

        :keyword  tags: A list of tags to associate with the node.
        :type     tags: ``list`` of ``str`` or ``None``

        :keyword  metadata: Metadata dictionary for instance.
        :type     metadata: ``dict`` or ``None``

        :keyword  description: The description of the node (instance).
        :type     description: ``str`` or ``None``

        :keyword  disks_gce_struct: Support for passing in the GCE-specific
                                    formatted disks[] structure. No attempt
                                    is made to ensure proper formatting of
                                    the disks[] structure. Using this
                                    structure obviates the need of using
                                    other disk params like 'boot_disk',
                                    etc. See the GCE docs for specific
                                    details.
        :type     disks_gce_struct: ``list`` or ``None``

        :keyword  nic_gce_struct: Support passing in the GCE-specific
                                  formatted networkInterfaces[] structure.
                                  No attempt is made to ensure proper
                                  formatting of the networkInterfaces[]
                                  data. Using this structure obviates the
                                  need of using 'external_ip' and
                                  'network'.  See the GCE docs for
                                  details.
        :type     nic_gce_struct: ``list`` or ``None``

        :type     labels: Labels dict for instance
        :type     labels: ``dict`` or ``None``

        :keyword  accelerator_type: Support for passing in the GCE-specifc
                                    accelerator type to request for the VM.
        :type     accelerator_type: :class:`GCEAcceleratorType` or ``None``

        :keyword  accelerator_count: Support for passing in the number of
                                     requested 'accelerator_type' accelerators
                                     attached to the VM. Will only pay attention
                                     to this field if 'accelerator_type' is not
                                     None.
        :type     accelerator_count: ``int`` or ``None``

        :keyword  disk_size: Specify size of the boot disk.
                             Integer in gigabytes.
        :type     disk_size: ``int`` or ``None``

        :return:  A dictionary formatted for use with the GCE API.
        :rtype:   ``dict``
        """
    instance_properties = {}
    if not image and (not source) and (not disks_gce_struct):
        raise ValueError("Missing root device or image. Must specify an 'image', source, or use the 'disks_gce_struct'.")
    if source and disks_gce_struct:
        raise ValueError("Cannot specify both 'source' and 'disks_gce_struct'. Use one or the other.")
    if disks_gce_struct:
        instance_properties['disks'] = disks_gce_struct
    else:
        disk_name = None
        device_name = None
        if source:
            disk_name = source.name
            device_name = source.name
            image = None
        instance_properties['disks'] = [self._build_disk_gce_struct(device_name, source=source, disk_type=disk_type, image=image, disk_name=disk_name, usage_type='PERSISTENT', mount_mode='READ_WRITE', auto_delete=disk_auto_delete, is_boot=True, use_selflinks=use_selflinks, disk_size=disk_size)]
    if nic_gce_struct is not None:
        if hasattr(external_ip, 'address'):
            raise ValueError("Cannot specify both a static IP address and 'nic_gce_struct'. Use one or the other.")
        if hasattr(network, 'name'):
            if network.name == 'default':
                network = None
            else:
                raise ValueError("Cannot specify both 'network' and 'nic_gce_struct'. Use one or the other.")
        instance_properties['networkInterfaces'] = nic_gce_struct
    else:
        instance_properties['networkInterfaces'] = [self._build_network_gce_struct(network=network, subnetwork=subnetwork, external_ip=external_ip, use_selflinks=True, internal_ip=internal_ip)]
    scheduling = self._build_scheduling_gce_struct(on_host_maintenance, automatic_restart, preemptible)
    if scheduling:
        instance_properties['scheduling'] = scheduling
    instance_properties['serviceAccounts'] = self._build_service_accounts_gce_list(service_accounts)
    if accelerator_type is not None:
        instance_properties['guestAccelerators'] = self._format_guest_accelerators(accelerator_type, accelerator_count)
    if description:
        instance_properties['description'] = str(description)
    if tags:
        instance_properties['tags'] = {'items': tags}
    if metadata:
        instance_properties['metadata'] = self._format_metadata(fingerprint='na', metadata=metadata)
    if labels:
        instance_properties['labels'] = labels
    if can_ip_forward:
        instance_properties['canIpForward'] = True
    instance_properties['machineType'] = self._get_selflink_or_name(obj=node_size, get_selflinks=use_selflinks, objname='size')
    return instance_properties