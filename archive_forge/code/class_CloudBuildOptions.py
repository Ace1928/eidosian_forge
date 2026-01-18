from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudBuildOptions(_messages.Message):
    """Options for the build operations performed as a part of the version
  deployment. Only applicable for App Engine flexible environment when
  creating a version using source code directly.

  Fields:
    appYamlPath: Path to the yaml file used in deployment, used to determine
      runtime configuration details.Required for flexible environment
      builds.See
      https://cloud.google.com/appengine/docs/standard/python/config/appref
      for more details.
    cloudBuildTimeout: The Cloud Build timeout used as part of any dependent
      builds performed by version creation. Defaults to 10 minutes.
  """
    appYamlPath = _messages.StringField(1)
    cloudBuildTimeout = _messages.StringField(2)