from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecommenderV1Recommendation(_messages.Message):
    """A recommendation along with a suggested action. E.g., a rightsizing
  recommendation for an underutilized VM, IAM role recommendations, etc

  Enums:
    PriorityValueValuesEnum: Recommendation's priority.

  Fields:
    additionalImpact: Optional set of additional impact that this
      recommendation may have when trying to optimize for the primary
      category. These may be positive or negative.
    associatedInsights: Insights that led to this recommendation.
    content: Content of the recommendation describing recommended changes to
      resources.
    description: Free-form human readable summary in English. The maximum
      length is 500 characters.
    etag: Fingerprint of the Recommendation. Provides optimistic locking when
      updating states.
    lastRefreshTime: Last time this recommendation was refreshed by the system
      that created it in the first place.
    name: Identifier. Name of recommendation.
    primaryImpact: The primary impact that this recommendation can have while
      trying to optimize for one category.
    priority: Recommendation's priority.
    recommenderSubtype: Contains an identifier for a subtype of
      recommendations produced for the same recommender. Subtype is a function
      of content and impact, meaning a new subtype might be added when
      significant changes to `content` or `primary_impact.category` are
      introduced. See the Recommenders section to see a list of subtypes for a
      given Recommender. Examples: For recommender =
      "google.iam.policy.Recommender", recommender_subtype can be one of
      "REMOVE_ROLE"/"REPLACE_ROLE"
    stateInfo: Information for state. Contains state and metadata.
    targetResources: Fully qualified resource names that this recommendation
      is targeting.
    xorGroupId: Corresponds to a mutually exclusive group ID within a
      recommender. A non-empty ID indicates that the recommendation belongs to
      a mutually exclusive group. This means that only one recommendation
      within the group is suggested to be applied.
  """

    class PriorityValueValuesEnum(_messages.Enum):
        """Recommendation's priority.

    Values:
      PRIORITY_UNSPECIFIED: Recommendation has unspecified priority.
      P4: Recommendation has P4 priority (lowest priority).
      P3: Recommendation has P3 priority (second lowest priority).
      P2: Recommendation has P2 priority (second highest priority).
      P1: Recommendation has P1 priority (highest priority).
    """
        PRIORITY_UNSPECIFIED = 0
        P4 = 1
        P3 = 2
        P2 = 3
        P1 = 4
    additionalImpact = _messages.MessageField('GoogleCloudRecommenderV1Impact', 1, repeated=True)
    associatedInsights = _messages.MessageField('GoogleCloudRecommenderV1RecommendationInsightReference', 2, repeated=True)
    content = _messages.MessageField('GoogleCloudRecommenderV1RecommendationContent', 3)
    description = _messages.StringField(4)
    etag = _messages.StringField(5)
    lastRefreshTime = _messages.StringField(6)
    name = _messages.StringField(7)
    primaryImpact = _messages.MessageField('GoogleCloudRecommenderV1Impact', 8)
    priority = _messages.EnumField('PriorityValueValuesEnum', 9)
    recommenderSubtype = _messages.StringField(10)
    stateInfo = _messages.MessageField('GoogleCloudRecommenderV1RecommendationStateInfo', 11)
    targetResources = _messages.StringField(12, repeated=True)
    xorGroupId = _messages.StringField(13)