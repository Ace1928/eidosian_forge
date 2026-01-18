from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1alpha2Impact(_messages.Message):
    """Contains the impact a recommendation can have for a given category.

  Enums:
    CategoryValueValuesEnum: Category that is being targeted.

  Fields:
    category: Category that is being targeted.
    costProjection: Use with CategoryType.COST
    reliabilityProjection: Use with CategoryType.RELIABILITY
    securityProjection: Use with CategoryType.SECURITY
    sustainabilityProjection: Use with CategoryType.SUSTAINABILITY
  """

    class CategoryValueValuesEnum(_messages.Enum):
        """Category that is being targeted.

    Values:
      CATEGORY_UNSPECIFIED: Default unspecified category. Don't use directly.
      COST: Indicates a potential increase or decrease in cost.
      SECURITY: Indicates a potential increase or decrease in security.
      PERFORMANCE: Indicates a potential increase or decrease in performance.
      MANAGEABILITY: Indicates a potential increase or decrease in
        manageability.
      SUSTAINABILITY: Indicates a potential increase or decrease in
        sustainability.
      RELIABILITY: Indicates a potential increase or decrease in reliability.
    """
        CATEGORY_UNSPECIFIED = 0
        COST = 1
        SECURITY = 2
        PERFORMANCE = 3
        MANAGEABILITY = 4
        SUSTAINABILITY = 5
        RELIABILITY = 6
    category = _messages.EnumField('CategoryValueValuesEnum', 1)
    costProjection = _messages.MessageField('GoogleCloudRecommenderV1alpha2CostProjection', 2)
    reliabilityProjection = _messages.MessageField('GoogleCloudRecommenderV1alpha2ReliabilityProjection', 3)
    securityProjection = _messages.MessageField('GoogleCloudRecommenderV1alpha2SecurityProjection', 4)
    sustainabilityProjection = _messages.MessageField('GoogleCloudRecommenderV1alpha2SustainabilityProjection', 5)