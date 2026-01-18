from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNpmPackagesResponse(_messages.Message):
    """The response from listing npm packages.

  Fields:
    nextPageToken: The token to retrieve the next page of artifacts, or empty
      if there are no more artifacts to return.
    npmPackages: The npm packages returned.
  """
    nextPageToken = _messages.StringField(1)
    npmPackages = _messages.MessageField('NpmPackage', 2, repeated=True)