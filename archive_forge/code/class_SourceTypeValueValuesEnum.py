from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceTypeValueValuesEnum(_messages.Enum):
    """Type of the model source.

    Values:
      MODEL_SOURCE_TYPE_UNSPECIFIED: Should not be used.
      AUTOML: The Model is uploaded by automl training pipeline.
      CUSTOM: The Model is uploaded by user or custom training pipeline.
      BQML: The Model is registered and sync'ed from BigQuery ML.
      CUSTOM_TEXT_EMBEDDING: The Model is uploaded by text embedding
        finetuning pipeline.
      MARKETPLACE: The Model is saved or tuned from Marketplace.
    """
    MODEL_SOURCE_TYPE_UNSPECIFIED = 0
    AUTOML = 1
    CUSTOM = 2
    BQML = 3
    CUSTOM_TEXT_EMBEDDING = 4
    MARKETPLACE = 5