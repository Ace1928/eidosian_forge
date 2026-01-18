from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaCounterScopeValueValuesEnum(_messages.Enum):
    """Scope of the quota decides how the quota counter gets applied and
    evaluate for quota violation. If the Scope is set as PROXY, then all the
    operations defined for the APIproduct that are associated with the same
    proxy will share the same quota counter set at the APIproduct level,
    making it a global counter at a proxy level. If the Scope is set as
    OPERATION, then each operations get the counter set at the API product
    dedicated, making it a local counter. Note that, the QuotaCounterScope
    applies only when an operation does not have dedicated quota set for
    itself.

    Values:
      QUOTA_COUNTER_SCOPE_UNSPECIFIED: When quota is not explicitly defined
        for each operation(REST/GraphQL), the limits set at product level will
        be used as a local counter for quota evaluation by all the operations,
        independent of proxy association.
      PROXY: When quota is not explicitly defined for each
        operation(REST/GraphQL), set at product level will be used as a global
        counter for quota evaluation by all the operations associated with a
        particular proxy.
      OPERATION: When quota is not explicitly defined for each
        operation(REST/GraphQL), the limits set at product level will be used
        as a local counter for quota evaluation by all the operations,
        independent of proxy association. This behavior mimics the same as
        QUOTA_COUNTER_SCOPE_UNSPECIFIED.
    """
    QUOTA_COUNTER_SCOPE_UNSPECIFIED = 0
    PROXY = 1
    OPERATION = 2