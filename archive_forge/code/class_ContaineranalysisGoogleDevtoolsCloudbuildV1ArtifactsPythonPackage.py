from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisGoogleDevtoolsCloudbuildV1ArtifactsPythonPackage(_messages.Message):
    """Python package to upload to Artifact Registry upon successful completion
  of all build steps. A package can encapsulate multiple objects to be
  uploaded to a single repository.

  Fields:
    paths: Path globs used to match files in the build's workspace. For
      Python/ Twine, this is usually `dist/*`, and sometimes additionally an
      `.asc` file.
    repository: Artifact Registry repository, in the form "https://$REGION-
      python.pkg.dev/$PROJECT/$REPOSITORY" Files in the workspace matching any
      path pattern will be uploaded to Artifact Registry with this location as
      a prefix.
  """
    paths = _messages.StringField(1, repeated=True)
    repository = _messages.StringField(2)