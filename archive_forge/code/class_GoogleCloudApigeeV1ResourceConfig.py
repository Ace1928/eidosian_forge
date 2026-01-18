from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ResourceConfig(_messages.Message):
    """A GoogleCloudApigeeV1ResourceConfig object.

  Fields:
    location: Location of the resource as a URI.
    name: Resource name in the following format: `organizations/{org}/environm
      ents/{env}/resourcefiles/{type}/{file}/revisions/{rev}` Only
      environment-scoped resource files are supported.
  """
    location = _messages.StringField(1)
    name = _messages.StringField(2)