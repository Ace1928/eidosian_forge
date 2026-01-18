from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpannerProjectsInstanceConfigsDeleteRequest(_messages.Message):
    """A SpannerProjectsInstanceConfigsDeleteRequest object.

  Fields:
    etag: Used for optimistic concurrency control as a way to help prevent
      simultaneous deletes of an instance config from overwriting each other.
      If not empty, the API only deletes the instance config when the etag
      provided matches the current status of the requested instance config.
      Otherwise, deletes the instance config without checking the current
      status of the requested instance config.
    name: Required. The name of the instance configuration to be deleted.
      Values are of the form `projects//instanceConfigs/`
    validateOnly: An option to validate, but not actually execute, a request,
      and provide the same response.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)