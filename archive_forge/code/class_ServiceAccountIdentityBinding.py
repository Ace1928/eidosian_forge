from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceAccountIdentityBinding(_messages.Message):
    """Represents a service account identity provider reference. A service
  account has at most one identity binding for the EAP. This is an alternative
  to service account keys and enables the service account to be configured to
  trust an external IDP through the provided identity binding.

  Fields:
    acceptanceFilter: A CEL expression that is evaluated to determine whether
      a credential should be accepted. To accept any credential, specify
      "true". See: https://github.com/google/cel-spec . This field supports a
      subset of the CEL functionality to select fields and evaluate boolean
      expressions based on the input (no functions or arithmetics). The values
      for input claims are available using `inclaim.attribute_name` or
      `inclaim[\\"attribute_name\\"]`. The values for output attributes
      calculated by the translator are available using
      `outclaim.attribute_name` or `outclaim[\\"attribute_name\\"]`.
    cel: A set of output attributes and corresponding input attribute
      expressions.
    name: The resource name of the service account identity binding in the
      following format `projects/{PROJECT_ID}/serviceAccounts/{ACCOUNT}/identi
      tyBindings/{BINDING}`.
    oidc: OIDC with discovery.
  """
    acceptanceFilter = _messages.StringField(1)
    cel = _messages.MessageField('AttributeTranslatorCEL', 2)
    name = _messages.StringField(3)
    oidc = _messages.MessageField('IDPReferenceOIDC', 4)