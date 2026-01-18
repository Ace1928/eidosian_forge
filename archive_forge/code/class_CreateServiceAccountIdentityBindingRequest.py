from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateServiceAccountIdentityBindingRequest(_messages.Message):
    """The service account identity binding create request.

  Fields:
    acceptanceFilter: A CEL expression that is evaluated to determine whether
      a credential should be accepted. To accept any credential, specify
      "true". See: https://github.com/google/cel-spec . The input claims are
      available using "inclaim[\\"attribute_name\\"]". The output attributes
      calculated by the translator are available using
      "outclaim[\\"attribute_name\\"]"
    cel: A set of output attributes and corresponding input attribute names.
    oidc: An OIDC reference with Discovery.
  """
    acceptanceFilter = _messages.StringField(1)
    cel = _messages.MessageField('AttributeTranslatorCEL', 2)
    oidc = _messages.MessageField('IDPReferenceOIDC', 3)