from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageVersion(_messages.Message):
    """Image Version information

  Fields:
    creationDisabled: Whether it is impossible to create an environment with
      the image version.
    imageVersionId: The string identifier of the ImageVersion, in the form:
      "composer-x.y.z-airflow-a.b.c"
    isDefault: Whether this is the default ImageVersion used by Composer
      during environment creation if no input ImageVersion is specified.
    releaseDate: The date of the version release.
    supportedPythonVersions: supported python versions
    upgradeDisabled: Whether it is impossible to upgrade an environment
      running with the image version.
  """
    creationDisabled = _messages.BooleanField(1)
    imageVersionId = _messages.StringField(2)
    isDefault = _messages.BooleanField(3)
    releaseDate = _messages.MessageField('Date', 4)
    supportedPythonVersions = _messages.StringField(5, repeated=True)
    upgradeDisabled = _messages.BooleanField(6)