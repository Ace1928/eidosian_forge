from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsGenerateRandomBytesRequest(_messages.Message):
    """A CloudkmsProjectsLocationsGenerateRandomBytesRequest object.

  Fields:
    generateRandomBytesRequest: A GenerateRandomBytesRequest resource to be
      passed as the request body.
    location: The project-specific location in which to generate random bytes.
      For example, "projects/my-project/locations/us-central1".
  """
    generateRandomBytesRequest = _messages.MessageField('GenerateRandomBytesRequest', 1)
    location = _messages.StringField(2, required=True)