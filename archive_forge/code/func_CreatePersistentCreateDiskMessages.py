from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import csek_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import instance_utils
from googlecloudsdk.api_lib.compute import kms_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.compute.instances import utils as instances_utils
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.compute import scope as compute_scopes
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
def CreatePersistentCreateDiskMessages(compute_client, resources, csek_keys, create_disks, project, location, scope, holder, enable_kms=False, enable_snapshots=False, container_mount_disk=None, enable_source_snapshot_csek=False, enable_image_csek=False, support_replica_zones=False, use_disk_type_uri=True, support_multi_writer=False, support_image_family_scope=False, enable_source_instant_snapshots=False, support_enable_confidential_compute=False):
    """Returns a list of AttachedDisk messages for newly creating disks.

  Args:
    compute_client: creates resources,
    resources: parser of resources,
    csek_keys: customer suplied encryption keys,
    create_disks: disk objects - contains following properties * name - the name
      of disk, * description - an optional description for the disk, * mode -
      'rw' (R/W), 'ro' (R/O) access mode, * disk-size - the size of the disk, *
      disk-type - the type of the disk (HDD or SSD), * image - the name of the
      image to initialize from, * image-csek-required - the name of the CSK
      protected image, * image-family - the image family name, * image-project -
      the project name that has the image, * auto-delete - whether disks is
      deleted when VM is deleted, * device-name - device name on VM, *
      source-snapshot - the snapshot to initialize from, *
      source-snapshot-csek-required - CSK protected snapshot, *
      source-instant-snapshot - the instant snapshot to initialize from, *
      disk-resource-policy - resource policies applied to disk. *
      enable_source_snapshot_csek - CSK file for snapshot, * enable_image_csek -
      CSK file for image
    project: Project of instance that will own the new disks.
    location: Location of the instance that will own the new disks.
    scope: Location type of the instance that will own the new disks.
    holder: Convenience class to hold lazy initialized client and resources.
    enable_kms: True if KMS keys are supported for the disk.
    enable_snapshots: True if snapshot initialization is supported for the disk.
    container_mount_disk: list of disks to be mounted to container, if any.
    enable_source_snapshot_csek: True if snapshot CSK files are enabled
    enable_image_csek: True if image CSK files are enabled
    support_replica_zones: True if we allow creation of regional disks
    use_disk_type_uri: True to use disk type URI, False if naked type.
    support_multi_writer: True if we allow multiple instances to write to disk.
    support_image_family_scope: True if the zonal image views are supported.
    enable_source_instant_snapshots: True if instant snapshot initialization is
      supported for the disk.
    support_enable_confidential_compute: True to use confidential mode for disk.

  Returns:
    list of API messages for attached disks
  """
    disks_messages = []
    messages = compute_client.messages
    compute = compute_client.apitools_client
    for disk in create_disks or []:
        name = disk.get('name')
        mode_value = disk.get('mode', 'rw')
        if mode_value == 'rw':
            mode = messages.AttachedDisk.ModeValueValuesEnum.READ_WRITE
        else:
            mode = messages.AttachedDisk.ModeValueValuesEnum.READ_ONLY
        auto_delete = disk.get('auto-delete', True)
        disk_size_gb = utils.BytesToGb(disk.get('size'))
        replica_zones = disk.get('replica-zones', [])
        disk_type = disk.get('type')
        if disk_type:
            if use_disk_type_uri:
                disk_type_ref = instance_utils.ParseDiskType(resources, disk_type, project, location, scope, replica_zone_cnt=len(replica_zones))
                disk_type = disk_type_ref.SelfLink()
        else:
            disk_type = None
        img = disk.get('image')
        img_family = disk.get('image-family')
        img_project = disk.get('image-project')
        image_family_scope = disk.get('image_family_scope')
        image_uri = None
        if img or img_family:
            image_expander = image_utils.ImageExpander(compute_client, resources)
            image_uri, _ = image_expander.ExpandImageFlag(user_project=project, image=img, image_family=img_family, image_project=img_project, return_image_resource=False, image_family_scope=image_family_scope, support_image_family_scope=support_image_family_scope)
        image_key = None
        disk_key = None
        if csek_keys:
            image_key = csek_utils.MaybeLookupKeyMessagesByUri(csek_keys, resources, [image_uri], compute)
            if name:
                disk_ref = resources.Parse(name, collection='compute.disks', params={'zone': location})
                disk_key = csek_utils.MaybeLookupKeyMessage(csek_keys, disk_ref, compute)
        if enable_kms:
            disk_key = kms_utils.MaybeGetKmsKeyFromDict(disk, messages, disk_key)
        initialize_params = messages.AttachedDiskInitializeParams(diskName=name, description=disk.get('description'), sourceImage=image_uri, diskSizeGb=disk_size_gb, diskType=disk_type, sourceImageEncryptionKey=image_key)
        if support_replica_zones and replica_zones:
            normalized_zones = []
            for zone in replica_zones:
                zone_ref = holder.resources.Parse(zone, collection='compute.zones', params={'project': project})
                normalized_zones.append(zone_ref.SelfLink())
            initialize_params.replicaZones = normalized_zones
        if enable_snapshots:
            snapshot_name = disk.get('source-snapshot')
            attached_snapshot_uri = instance_utils.ResolveSnapshotURI(snapshot=snapshot_name, user_project=project, resource_parser=resources)
            if attached_snapshot_uri:
                initialize_params.sourceImage = None
                initialize_params.sourceSnapshot = attached_snapshot_uri
        policies = disk.get('disk-resource-policy')
        if policies:
            initialize_params.resourcePolicies = policies
        if enable_image_csek:
            image_key_file = disk.get('image_csek')
            if image_key_file:
                initialize_params.imageKeyFile = image_key_file
        if enable_source_snapshot_csek:
            snapshot_key_file = disk.get('source_snapshot_csek')
            if snapshot_key_file:
                initialize_params.snapshotKeyFile = snapshot_key_file
        if enable_source_instant_snapshots:
            instant_snapshot_name = disk.get('source-instant-snapshot')
            attached_instant_snapshot_uri = instance_utils.ResolveInstantSnapshotURI(user_project=project, instant_snapshot=instant_snapshot_name, resource_parser=resources)
            if attached_instant_snapshot_uri:
                initialize_params.sourceImage = None
                initialize_params.sourceSnapshot = None
                initialize_params.sourceInstantSnapshot = attached_instant_snapshot_uri
        boot = disk.get('boot', False)
        multi_writer = disk.get('multi-writer')
        if support_multi_writer and multi_writer:
            initialize_params.multiWriter = True
        enable_confidential_compute = disk.get('confidential-compute')
        if support_enable_confidential_compute and enable_confidential_compute:
            initialize_params.enableConfidentialCompute = True
        provisioned_iops = disk.get('provisioned-iops')
        if provisioned_iops:
            initialize_params.provisionedIops = provisioned_iops
        provisioned_throughput = disk.get('provisioned-throughput')
        if provisioned_throughput:
            initialize_params.provisionedThroughput = provisioned_throughput
        storage_pool = disk.get('storage-pool')
        if storage_pool:
            storage_pool_ref = instance_utils.ParseStoragePool(resources, storage_pool, project, location)
            storage_pool_uri = storage_pool_ref.SelfLink()
            initialize_params.storagePool = storage_pool_uri
        disk_architecture = disk.get('architecture')
        if disk_architecture:
            initialize_params.architecture = messages.AttachedDiskInitializeParams.ArchitectureValueValuesEnum(disk_architecture)
        device_name = instance_utils.GetDiskDeviceName(disk, name, container_mount_disk)
        create_disk = messages.AttachedDisk(autoDelete=auto_delete, boot=boot, deviceName=device_name, initializeParams=initialize_params, mode=mode, type=messages.AttachedDisk.TypeValueValuesEnum.PERSISTENT, diskEncryptionKey=disk_key)
        if boot:
            disks_messages = [create_disk] + disks_messages
        else:
            disks_messages.append(create_disk)
    return disks_messages