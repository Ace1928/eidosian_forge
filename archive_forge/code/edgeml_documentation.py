from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding_helper
from googlecloudsdk.api_lib.edgeml import operations
from googlecloudsdk.api_lib.edgeml import util
from googlecloudsdk.core import properties
Converts Tensorflow SavedModel to TFLite model.

    Args:
      source: str, GCS URI to an input SavedModel archive
      destination: str, GCS URI to an output tflite object. If not provided,
        for source filename `model[/saved_model.(pb|pbtxt)]` this will be
        set to `model.tflite`.

    Returns:
      (ConvertModelResponse, output object URI) on the finish of
      convert operation.

    Raises:
      LongrunningError: when long running operation fails.
    