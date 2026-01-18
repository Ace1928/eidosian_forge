from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TopologyManager(_messages.Message):
    """TopologyManager defines the configuration options for Topology Manager
  feature. See https://kubernetes.io/docs/tasks/administer-cluster/topology-
  manager/

  Fields:
    policy: Configures the strategy for resource alignment. Allowed values
      are: * none: the default policy, and does not perform any topology
      alignment. * restricted: the topology manager stores the preferred NUMA
      node affinity for the container, and will reject the pod if the affinity
      if not preferred. * best-effort: the topology manager stores the
      preferred NUMA node affinity for the container. If the affinity is not
      preferred, the topology manager will admit the pod to the node anyway. *
      single-numa-node: the topology manager determines if the single NUMA
      node affinity is possible. If it is, Topology Manager will store this
      and the Hint Providers can then use this information when making the
      resource allocation decision. If, however, this is not possible then the
      Topology Manager will reject the pod from the node. This will result in
      a pod in a Terminated state with a pod admission failure. The default
      policy value is 'none' if unspecified. Details about each strategy can
      be found [here](https://kubernetes.io/docs/tasks/administer-
      cluster/topology-manager/#topology-manager-policies).
    scope: The Topology Manager aligns resources in following scopes: *
      container * pod The default scope is 'container' if unspecified. See
      https://kubernetes.io/docs/tasks/administer-cluster/topology-
      manager/#topology-manager-scopes
  """
    policy = _messages.StringField(1)
    scope = _messages.StringField(2)