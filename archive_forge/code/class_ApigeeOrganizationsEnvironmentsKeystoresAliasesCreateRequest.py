from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsKeystoresAliasesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsKeystoresAliasesCreateRequest object.

  Fields:
    _password: DEPRECATED: For improved security, specify the password in the
      request body instead of using the query parameter. To specify the
      password in the request body, set `Content-type: multipart/form-data`
      part with name `password`. Password for the private key file, if
      required.
    alias: Alias for the key/certificate pair. Values must match the regular
      expression `[\\w\\s-.]{1,255}`. This must be provided for all formats
      except `selfsignedcert`; self-signed certs may specify the alias in
      either this parameter or the JSON body.
    format: Required. Format of the data. Valid values include:
      `selfsignedcert`, `keycertfile`, or `pkcs12`
    googleApiHttpBody: A GoogleApiHttpBody resource to be passed as the
      request body.
    ignoreExpiryValidation: Flag that specifies whether to ignore expiry
      validation. If set to `true`, no expiry validation will be performed.
    ignoreNewlineValidation: Flag that specifies whether to ignore newline
      validation. If set to `true`, no error is thrown when the file contains
      a certificate chain with no newline between each certificate. Defaults
      to `false`.
    parent: Required. Name of the keystore. Use the following format in your
      request: `organizations/{org}/environments/{env}/keystores/{keystore}`.
  """
    _password = _messages.StringField(1)
    alias = _messages.StringField(2)
    format = _messages.StringField(3)
    googleApiHttpBody = _messages.MessageField('GoogleApiHttpBody', 4)
    ignoreExpiryValidation = _messages.BooleanField(5)
    ignoreNewlineValidation = _messages.BooleanField(6)
    parent = _messages.StringField(7, required=True)