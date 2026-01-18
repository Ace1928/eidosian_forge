from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IPAllocationPolicy(_messages.Message):
    """Configuration for controlling how IPs are allocated in the GKE cluster
  running the Apache Airflow software.

  Fields:
    clusterIpv4CidrBlock: Optional. The IP address range used to allocate IP
      addresses to pods in the GKE cluster. For Cloud Composer environments in
      versions composer-1.*.*-airflow-*.*.*, this field is applicable only
      when `use_ip_aliases` is true. Set to blank to have GKE choose a range
      with the default size. Set to /netmask (e.g. `/14`) to have GKE choose a
      range with a specific netmask. Set to a
      [CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
      notation (e.g. `10.96.0.0/14`) from the RFC-1918 private networks (e.g.
      `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) to pick a specific
      range to use.
    clusterSecondaryRangeName: Optional. The name of the GKE cluster's
      secondary range used to allocate IP addresses to pods. For Cloud
      Composer environments in versions composer-1.*.*-airflow-*.*.*, this
      field is applicable only when `use_ip_aliases` is true.
    servicesIpv4CidrBlock: Optional. The IP address range of the services IP
      addresses in this GKE cluster. For Cloud Composer environments in
      versions composer-1.*.*-airflow-*.*.*, this field is applicable only
      when `use_ip_aliases` is true. Set to blank to have GKE choose a range
      with the default size. Set to /netmask (e.g. `/14`) to have GKE choose a
      range with a specific netmask. Set to a
      [CIDR](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing)
      notation (e.g. `10.96.0.0/14`) from the RFC-1918 private networks (e.g.
      `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`) to pick a specific
      range to use.
    servicesSecondaryRangeName: Optional. The name of the services' secondary
      range used to allocate IP addresses to the GKE cluster. For Cloud
      Composer environments in versions composer-1.*.*-airflow-*.*.*, this
      field is applicable only when `use_ip_aliases` is true.
    useIpAliases: Optional. Whether or not to enable Alias IPs in the GKE
      cluster. If `true`, a VPC-native cluster is created. This field is only
      supported for Cloud Composer environments in versions
      composer-1.*.*-airflow-*.*.*. Environments in newer versions always use
      VPC-native GKE clusters.
  """
    clusterIpv4CidrBlock = _messages.StringField(1)
    clusterSecondaryRangeName = _messages.StringField(2)
    servicesIpv4CidrBlock = _messages.StringField(3)
    servicesSecondaryRangeName = _messages.StringField(4)
    useIpAliases = _messages.BooleanField(5)