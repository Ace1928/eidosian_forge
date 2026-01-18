from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MlProjectsModelsVersionsPatchRequest(_messages.Message):
    """A MlProjectsModelsVersionsPatchRequest object.

  Fields:
    googleCloudMlV1Version: A GoogleCloudMlV1Version resource to be passed as
      the request body.
    name: Required. The name of the model.
    updateMask: Required. Specifies the path, relative to `Version`, of the
      field to update. Must be present and non-empty. For example, to change
      the description of a version to "foo", the `update_mask` parameter would
      be specified as `description`, and the `PATCH` request body would
      specify the new value, as follows: ``` { "description": "foo" } ```
      Currently the only supported update mask fields are `description`,
      `requestLoggingConfig`, `autoScaling.minNodes`, and
      `manualScaling.nodes`. However, you can only update
      `manualScaling.nodes` if the version uses a [Compute Engine (N1) machine
      type](/ml-engine/docs/machine-types-online-prediction).
  """
    googleCloudMlV1Version = _messages.MessageField('GoogleCloudMlV1Version', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)