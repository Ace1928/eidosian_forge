from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDatastoreAdminV1PrepareStepDetails(_messages.Message):
    """Details for the `PREPARE` step.

  Enums:
    ConcurrencyModeValueValuesEnum: The concurrency mode this database will
      use when it reaches the `REDIRECT_WRITES` step.

  Fields:
    concurrencyMode: The concurrency mode this database will use when it
      reaches the `REDIRECT_WRITES` step.
  """

    class ConcurrencyModeValueValuesEnum(_messages.Enum):
        """The concurrency mode this database will use when it reaches the
    `REDIRECT_WRITES` step.

    Values:
      CONCURRENCY_MODE_UNSPECIFIED: Unspecified.
      PESSIMISTIC: Pessimistic concurrency.
      OPTIMISTIC: Optimistic concurrency.
      OPTIMISTIC_WITH_ENTITY_GROUPS: Optimistic concurrency with entity
        groups.
    """
        CONCURRENCY_MODE_UNSPECIFIED = 0
        PESSIMISTIC = 1
        OPTIMISTIC = 2
        OPTIMISTIC_WITH_ENTITY_GROUPS = 3
    concurrencyMode = _messages.EnumField('ConcurrencyModeValueValuesEnum', 1)