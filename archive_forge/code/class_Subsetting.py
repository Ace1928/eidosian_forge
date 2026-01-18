from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Subsetting(_messages.Message):
    """Subsetting configuration for this BackendService. Currently this is
  applicable only for Internal TCP/UDP load balancing, Internal HTTP(S) load
  balancing and Traffic Director.

  Enums:
    PolicyValueValuesEnum:

  Fields:
    policy: A PolicyValueValuesEnum attribute.
    subsetSize: The number of backends per backend group assigned to each
      proxy instance or each service mesh client. An input parameter to the
      `CONSISTENT_HASH_SUBSETTING` algorithm. Can only be set if `policy` is
      set to `CONSISTENT_HASH_SUBSETTING`. Can only be set if load balancing
      scheme is `INTERNAL_MANAGED` or `INTERNAL_SELF_MANAGED`. `subset_size`
      is optional for Internal HTTP(S) load balancing and required for Traffic
      Director. If you do not provide this value, Cloud Load Balancing will
      calculate it dynamically to optimize the number of proxies/clients
      visible to each backend and vice versa. Must be greater than 0. If
      `subset_size` is larger than the number of backends/endpoints, then
      subsetting is disabled.
  """

    class PolicyValueValuesEnum(_messages.Enum):
        """PolicyValueValuesEnum enum type.

    Values:
      CONSISTENT_HASH_SUBSETTING: Subsetting based on consistent hashing. For
        Traffic Director, the number of backends per backend group (the subset
        size) is based on the `subset_size` parameter. For Internal HTTP(S)
        load balancing, the number of backends per backend group (the subset
        size) is dynamically adjusted in two cases: - As the number of proxy
        instances participating in Internal HTTP(S) load balancing increases,
        the subset size decreases. - When the total number of backends in a
        network exceeds the capacity of a single proxy instance, subset sizes
        are reduced automatically for each service that has backend subsetting
        enabled.
      NONE: No Subsetting. Clients may open connections and send traffic to
        all backends of this backend service. This can lead to performance
        issues if there is substantial imbalance in the count of clients and
        backends.
    """
        CONSISTENT_HASH_SUBSETTING = 0
        NONE = 1
    policy = _messages.EnumField('PolicyValueValuesEnum', 1)
    subsetSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)