from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1FeatureViewVectorSearchConfig(_messages.Message):
    """Deprecated. Use IndexConfig instead.

  Enums:
    DistanceMeasureTypeValueValuesEnum: Optional. The distance measure used in
      nearest neighbor search.

  Fields:
    bruteForceConfig: Optional. Configuration options for using brute force
      search, which simply implements the standard linear search in the
      database for each query. It is primarily meant for benchmarking and to
      generate the ground truth for approximate search.
    crowdingColumn: Optional. Column of crowding. This column contains
      crowding attribute which is a constraint on a neighbor list produced by
      FeatureOnlineStoreService.SearchNearestEntities to diversify search
      results. If NearestNeighborQuery.per_crowding_attribute_neighbor_count
      is set to K in SearchNearestEntitiesRequest, it's guaranteed that no
      more than K entities of the same crowding attribute are returned in the
      response.
    distanceMeasureType: Optional. The distance measure used in nearest
      neighbor search.
    embeddingColumn: Optional. Column of embedding. This column contains the
      source data to create index for vector search. embedding_column must be
      set when using vector search.
    embeddingDimension: Optional. The number of dimensions of the input
      embedding.
    filterColumns: Optional. Columns of features that're used to filter vector
      search results.
    treeAhConfig: Optional. Configuration options for the tree-AH algorithm
      (Shallow tree + Asymmetric Hashing). Please refer to this paper for more
      details: https://arxiv.org/abs/1908.10396
  """

    class DistanceMeasureTypeValueValuesEnum(_messages.Enum):
        """Optional. The distance measure used in nearest neighbor search.

    Values:
      DISTANCE_MEASURE_TYPE_UNSPECIFIED: Should not be set.
      SQUARED_L2_DISTANCE: Euclidean (L_2) Distance.
      COSINE_DISTANCE: Cosine Distance. Defined as 1 - cosine similarity. We
        strongly suggest using DOT_PRODUCT_DISTANCE + UNIT_L2_NORM instead of
        COSINE distance. Our algorithms have been more optimized for
        DOT_PRODUCT distance which, when combined with UNIT_L2_NORM, is
        mathematically equivalent to COSINE distance and results in the same
        ranking.
      DOT_PRODUCT_DISTANCE: Dot Product Distance. Defined as a negative of the
        dot product.
    """
        DISTANCE_MEASURE_TYPE_UNSPECIFIED = 0
        SQUARED_L2_DISTANCE = 1
        COSINE_DISTANCE = 2
        DOT_PRODUCT_DISTANCE = 3
    bruteForceConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewVectorSearchConfigBruteForceConfig', 1)
    crowdingColumn = _messages.StringField(2)
    distanceMeasureType = _messages.EnumField('DistanceMeasureTypeValueValuesEnum', 3)
    embeddingColumn = _messages.StringField(4)
    embeddingDimension = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    filterColumns = _messages.StringField(6, repeated=True)
    treeAhConfig = _messages.MessageField('GoogleCloudAiplatformV1beta1FeatureViewVectorSearchConfigTreeAHConfig', 7)