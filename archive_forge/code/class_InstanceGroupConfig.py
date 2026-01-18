from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupConfig(_messages.Message):
    """The config settings for Compute Engine resources in an instance group,
  such as a master or worker group.

  Enums:
    PreemptibilityValueValuesEnum: Optional. Specifies the preemptibility of
      the instance group.The default value for master and worker groups is
      NON_PREEMPTIBLE. This default cannot be changed.The default value for
      secondary instances is PREEMPTIBLE.

  Fields:
    accelerators: Optional. The Compute Engine accelerator configuration for
      these instances.
    diskConfig: Optional. Disk option config settings.
    imageUri: Optional. The Compute Engine image resource used for cluster
      instances.The URI can represent an image or image family.Image examples:
      https://www.googleapis.com/compute/v1/projects/[project_id]/global/image
      s/[image-id] projects/[project_id]/global/images/[image-id] image-
      idImage family examples. Dataproc will use the most recent image from
      the family: https://www.googleapis.com/compute/v1/projects/[project_id]/
      global/images/family/[custom-image-family-name]
      projects/[project_id]/global/images/family/[custom-image-family-name]If
      the URI is unspecified, it will be inferred from
      SoftwareConfig.image_version or the system default.
    instanceFlexibilityPolicy: Optional. Instance flexibility Policy allowing
      a mixture of VM shapes and provisioning models.
    instanceNames: Output only. The list of instance names. Dataproc derives
      the names from cluster_name, num_instances, and the instance group.
    instanceReferences: Output only. List of references to Compute Engine
      instances.
    isPreemptible: Output only. Specifies that this instance group contains
      preemptible instances.
    machineTypeUri: Optional. The Compute Engine machine type used for cluster
      instances.A full URL, partial URI, or short name are valid. Examples: ht
      tps://www.googleapis.com/compute/v1/projects/[project_id]/zones/[zone]/m
      achineTypes/n1-standard-2
      projects/[project_id]/zones/[zone]/machineTypes/n1-standard-2
      n1-standard-2Auto Zone Exception: If you are using the Dataproc Auto
      Zone Placement
      (https://cloud.google.com/dataproc/docs/concepts/configuring-
      clusters/auto-zone#using_auto_zone_placement) feature, you must use the
      short name of the machine type resource, for example, n1-standard-2.
    managedGroupConfig: Output only. The config for Compute Engine Instance
      Group Manager that manages this group. This is only used for preemptible
      instance groups.
    minCpuPlatform: Optional. Specifies the minimum cpu platform for the
      Instance Group. See Dataproc -> Minimum CPU Platform
      (https://cloud.google.com/dataproc/docs/concepts/compute/dataproc-min-
      cpu).
    minNumInstances: Optional. The minimum number of primary worker instances
      to create. If min_num_instances is set, cluster creation will succeed if
      the number of primary workers created is at least equal to the
      min_num_instances number.Example: Cluster creation request with
      num_instances = 5 and min_num_instances = 3: If 4 VMs are created and 1
      instance fails, the failed VM is deleted. The cluster is resized to 4
      instances and placed in a RUNNING state. If 2 instances are created and
      3 instances fail, the cluster in placed in an ERROR state. The failed
      VMs are not deleted.
    numInstances: Optional. The number of VM instances in the instance group.
      For HA cluster master_config groups, must be set to 3. For standard
      cluster master_config groups, must be set to 1.
    preemptibility: Optional. Specifies the preemptibility of the instance
      group.The default value for master and worker groups is NON_PREEMPTIBLE.
      This default cannot be changed.The default value for secondary instances
      is PREEMPTIBLE.
    startupConfig: Optional. Configuration to handle the startup of instances
      during cluster create and update process.
  """

    class PreemptibilityValueValuesEnum(_messages.Enum):
        """Optional. Specifies the preemptibility of the instance group.The
    default value for master and worker groups is NON_PREEMPTIBLE. This
    default cannot be changed.The default value for secondary instances is
    PREEMPTIBLE.

    Values:
      PREEMPTIBILITY_UNSPECIFIED: Preemptibility is unspecified, the system
        will choose the appropriate setting for each instance group.
      NON_PREEMPTIBLE: Instances are non-preemptible.This option is allowed
        for all instance groups and is the only valid value for Master and
        Worker instance groups.
      PREEMPTIBLE: Instances are preemptible
        (https://cloud.google.com/compute/docs/instances/preemptible).This
        option is allowed only for secondary worker
        (https://cloud.google.com/dataproc/docs/concepts/compute/secondary-
        vms) groups.
      SPOT: Instances are Spot VMs
        (https://cloud.google.com/compute/docs/instances/spot).This option is
        allowed only for secondary worker
        (https://cloud.google.com/dataproc/docs/concepts/compute/secondary-
        vms) groups. Spot VMs are the latest version of preemptible VMs
        (https://cloud.google.com/compute/docs/instances/preemptible), and
        provide additional features.
    """
        PREEMPTIBILITY_UNSPECIFIED = 0
        NON_PREEMPTIBLE = 1
        PREEMPTIBLE = 2
        SPOT = 3
    accelerators = _messages.MessageField('AcceleratorConfig', 1, repeated=True)
    diskConfig = _messages.MessageField('DiskConfig', 2)
    imageUri = _messages.StringField(3)
    instanceFlexibilityPolicy = _messages.MessageField('InstanceFlexibilityPolicy', 4)
    instanceNames = _messages.StringField(5, repeated=True)
    instanceReferences = _messages.MessageField('InstanceReference', 6, repeated=True)
    isPreemptible = _messages.BooleanField(7)
    machineTypeUri = _messages.StringField(8)
    managedGroupConfig = _messages.MessageField('ManagedGroupConfig', 9)
    minCpuPlatform = _messages.StringField(10)
    minNumInstances = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    numInstances = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    preemptibility = _messages.EnumField('PreemptibilityValueValuesEnum', 13)
    startupConfig = _messages.MessageField('StartupConfig', 14)