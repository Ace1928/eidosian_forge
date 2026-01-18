from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CDNPolicyAddSignaturesOptions(_messages.Message):
    """The configuration options for adding signatures to responses.

  Enums:
    ActionsValueListEntryValuesEnum:

  Fields:
    actions: Required. The actions to take to add signatures to responses. You
      must specify exactly one action.
    copiedParameters: Optional. The parameters to copy from the verified token
      to the generated token. Only the following parameters can be copied: *
      `PathGlobs` * `paths` * `acl` * `URLPrefix` * `IPRanges` * `SessionID` *
      `id` * `Data` * `data` * `payload` * `Headers` You can specify up to 6
      parameters to copy. A given parameter is be copied only if the parameter
      exists in the verified token. Parameter names are matched exactly as
      specified. The order of the parameters does not matter. Duplicates are
      not allowed. This field can only be specified when the `GENERATE_COOKIE`
      or `GENERATE_TOKEN_HLS_COOKIELESS` actions are specified.
    keyset: Optional. The keyset to use for signature generation. The
      following are both valid paths to an EdgeCacheKeyset resource: *
      `projects/project/locations/global/edgeCacheKeysets/yourKeyset` *
      `yourKeyset` This must be specified when the `GENERATE_COOKIE` or
      `GENERATE_TOKEN_HLS_COOKIELESS` actions are specified. This field can
      not be specified otherwise.
    tokenQueryParameter: Optional. The query parameter in which to put the
      generated token. If not specified, defaults to `edge-cache-token`. If
      specified, the name must be 1-64 characters long and match the regular
      expression `[a-zA-Z]([a-zA-Z0-9_-])*` which means the first character
      must be a letter, and all following characters must be a dash,
      underscore, letter or digit. This field can only be set when the
      `GENERATE_TOKEN_HLS_COOKIELESS` or `PROPAGATE_TOKEN_HLS_COOKIELESS`
      actions are specified.
    tokenTtl: Optional. The duration the token is valid for starting from the
      moment the token is first generated. Defaults to `86400s` (1 day). The
      TTL must be >= 0 and <= 604,800 seconds (1 week). This field can only be
      specified when the `GENERATE_COOKIE` or `GENERATE_TOKEN_HLS_COOKIELESS`
      actions are specified.
  """

    class ActionsValueListEntryValuesEnum(_messages.Enum):
        """ActionsValueListEntryValuesEnum enum type.

    Values:
      SIGNATURE_ACTION_UNSPECIFIED: It is an error to specify `UNSPECIFIED`.
      GENERATE_COOKIE: Generate a new signed request cookie and return the
        cookie in a Set-Cookie header of the response. This action cannot be
        combined with the `PROPAGATE_TOKEN_HLS_COOKIELESS` action.
      GENERATE_TOKEN_HLS_COOKIELESS: Generate a new signed request
        authentication token and return the new token by manipulating URLs in
        an HTTP Live Stream (HLS) playlist. This action cannot be combined
        with the `PROPAGATE_TOKEN_HLS_COOKIELESS` action.
      PROPAGATE_TOKEN_HLS_COOKIELESS: Copy the authentication token used in
        the request to the URLs in an HTTP Live Stream (HLS) playlist. This
        action cannot be combined with either the `GENERATE_COOKIE` action or
        the `GENERATE_TOKEN_HLS_COOKIELESS` action.
    """
        SIGNATURE_ACTION_UNSPECIFIED = 0
        GENERATE_COOKIE = 1
        GENERATE_TOKEN_HLS_COOKIELESS = 2
        PROPAGATE_TOKEN_HLS_COOKIELESS = 3
    actions = _messages.EnumField('ActionsValueListEntryValuesEnum', 1, repeated=True)
    copiedParameters = _messages.StringField(2, repeated=True)
    keyset = _messages.StringField(3)
    tokenQueryParameter = _messages.StringField(4)
    tokenTtl = _messages.StringField(5)