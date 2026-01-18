from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DefaultServiceAccount(_messages.Message):
    """The default service account used for `Builds`.

  Fields:
    name: Identifier. Format:
      `projects/{project}/locations/{location}/defaultServiceAccount
    serviceAccountEmail: Output only. The email address of the service account
      identity that will be used for a build by default. This is returned in
      the format `projects/{project}/serviceAccounts/{service_account}` where
      `{service_account}` could be the legacy Cloud Build SA, in the format
      [PROJECT_NUMBER]@cloudbuild.gserviceaccount.com or the Compute SA, in
      the format [PROJECT_NUMBER]-compute@developer.gserviceaccount.com. If no
      service account will be used by default, this will be empty.
  """
    name = _messages.StringField(1)
    serviceAccountEmail = _messages.StringField(2)