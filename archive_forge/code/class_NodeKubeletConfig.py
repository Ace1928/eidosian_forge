from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeKubeletConfig(_messages.Message):
    """Node kubelet configs.

  Fields:
    cpuCfsQuota: Enable CPU CFS quota enforcement for containers that specify
      CPU limits. This option is enabled by default which makes kubelet use
      CFS quota (https://www.kernel.org/doc/Documentation/scheduler/sched-
      bwc.txt) to enforce container CPU limits. Otherwise, CPU limits will not
      be enforced at all. Disable this option to mitigate CPU throttling
      problems while still having your pods to be in Guaranteed QoS class by
      specifying the CPU limits. The default value is 'true' if unspecified.
    cpuCfsQuotaPeriod: Set the CPU CFS quota period value 'cpu.cfs_period_us'.
      The string must be a sequence of decimal numbers, each with optional
      fraction and a unit suffix, such as "300ms". Valid time units are "ns",
      "us" (or "\\xb5s"), "ms", "s", "m", "h". The value must be a positive
      duration.
    cpuManagerPolicy: Control the CPU management policy on the node. See
      https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-
      policies/ The following values are allowed. * "none": the default, which
      represents the existing scheduling behavior. * "static": allows pods
      with certain resource characteristics to be granted increased CPU
      affinity and exclusivity on the node. The default value is 'none' if
      unspecified.
    insecureKubeletReadonlyPortEnabled: Enable or disable Kubelet read only
      port.
    memoryManager: Optional. Controls NUMA-aware Memory Manager configuration
      on the node. For more information, see:
      https://kubernetes.io/docs/tasks/administer-cluster/memory-manager/
    podPidsLimit: Set the Pod PID limits. See
      https://kubernetes.io/docs/concepts/policy/pid-limiting/#pod-pid-limits
      Controls the maximum number of processes allowed to run in a pod. The
      value must be greater than or equal to 1024 and less than 4194304.
    topologyManager: Optional. Controls Topology Manager configuration on the
      node. For more information, see:
      https://kubernetes.io/docs/tasks/administer-cluster/topology-manager/
  """
    cpuCfsQuota = _messages.BooleanField(1)
    cpuCfsQuotaPeriod = _messages.StringField(2)
    cpuManagerPolicy = _messages.StringField(3)
    insecureKubeletReadonlyPortEnabled = _messages.BooleanField(4)
    memoryManager = _messages.MessageField('MemoryManager', 5)
    podPidsLimit = _messages.IntegerField(6)
    topologyManager = _messages.MessageField('TopologyManager', 7)