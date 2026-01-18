from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeregistryProjectsLocationsApisVersionsSpecsTagRevisionRequest(_messages.Message):
    """A ApigeeregistryProjectsLocationsApisVersionsSpecsTagRevisionRequest
  object.

  Fields:
    name: Required. The name of the spec to be tagged, including the revision
      ID is optional. If a revision is not specified, it will tag the latest
      revision.
    tagApiSpecRevisionRequest: A TagApiSpecRevisionRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    tagApiSpecRevisionRequest = _messages.MessageField('TagApiSpecRevisionRequest', 2)