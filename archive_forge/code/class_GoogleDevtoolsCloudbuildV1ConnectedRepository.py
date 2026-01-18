from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1ConnectedRepository(_messages.Message):
    """Location of the source in a 2nd-gen Google Cloud Build repository
  resource.

  Fields:
    dir: Directory, relative to the source root, in which to run the build.
    repository: Required. Name of the Google Cloud Build repository, formatted
      as `projects/*/locations/*/connections/*/repositories/*`.
    revision: The revision to fetch from the Git repository such as a branch,
      a tag, a commit SHA, or any Git ref.
  """
    dir = _messages.StringField(1)
    repository = _messages.StringField(2)
    revision = _messages.StringField(3)