from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildOauthGetRegistrationRequest(_messages.Message):
    """A CloudbuildOauthGetRegistrationRequest object.

  Enums:
    NamespaceValueValuesEnum: Required. The namespace that the credential
      belongs to.

  Fields:
    authUser: For users who are logged in using multiple accounts, specify the
      auth user parameter so that the registration url redirects back to the
      cloud console using the proper account.
    csesidx: Optional. For users who use byoid, specify the csesidx parameter
      so that the registration url redirects back to the cloud console using
      the proper account.
    githubEnterpriseConfig: Optional. The full resource name of the github
      enterprise resource if applicable.
    hostUrl: Required. The host url that the oauth credentials are associated
      with. For GitHub, this would be "https://github.com". For
      GitHubEnterprise, this would be the host name of their github enterprise
      instance.
    namespace: Required. The namespace that the credential belongs to.
  """

    class NamespaceValueValuesEnum(_messages.Enum):
        """Required. The namespace that the credential belongs to.

    Values:
      NAMESPACE_UNSPECIFIED: The default namespace.
      GITHUB_ENTERPRISE: A credential to be used with GitHub enterprise.
    """
        NAMESPACE_UNSPECIFIED = 0
        GITHUB_ENTERPRISE = 1
    authUser = _messages.StringField(1)
    csesidx = _messages.StringField(2)
    githubEnterpriseConfig = _messages.StringField(3)
    hostUrl = _messages.StringField(4)
    namespace = _messages.EnumField('NamespaceValueValuesEnum', 5)