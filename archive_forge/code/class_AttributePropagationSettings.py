from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttributePropagationSettings(_messages.Message):
    """Configuration for propagating attributes to applications protected by
  IAP.

  Enums:
    OutputCredentialsValueListEntryValuesEnum:

  Fields:
    enable: Whether the provided attribute propagation settings should be
      evaluated on user requests. If set to true, attributes returned from the
      expression will be propagated in the set output credentials.
    expression: Raw string CEL expression. Must return a list of attributes. A
      maximum of 45 attributes can be selected. Expressions can select
      different attribute types from `attributes`:
      `attributes.saml_attributes`, `attributes.iap_attributes`. The following
      functions are supported: - filter `.filter(, )`: Returns a subset of ``
      where `` is true for every item. - in ` in `: Returns true if ``
      contains ``. - selectByName `.selectByName()`: Returns the attribute in
      `` with the given `` name, otherwise returns empty. - emitAs
      `.emitAs()`: Sets the `` name field to the given `` for propagation in
      selected output credentials. - strict `.strict()`: Ignores the `x-goog-
      iap-attr-` prefix for the provided `` when propagating with the `HEADER`
      output credential, such as request headers. - append `.append()` OR
      `.append()`: Appends the provided `` or `` to the end of ``. Example
      expression: `attributes.saml_attributes.filter(x, x.name in ['test']).ap
      pend(attributes.iap_attributes.selectByName('exact').emitAs('custom').st
      rict())`
    outputCredentials: Which output credentials attributes selected by the CEL
      expression should be propagated in. All attributes will be fully
      duplicated in each selected output credential.
  """

    class OutputCredentialsValueListEntryValuesEnum(_messages.Enum):
        """OutputCredentialsValueListEntryValuesEnum enum type.

    Values:
      OUTPUT_CREDENTIALS_UNSPECIFIED: An output credential is required.
      HEADER: Propagate attributes in the headers with "x-goog-iap-attr-"
        prefix.
      JWT: Propagate attributes in the JWT of the form: `"additional_claims":
        { "my_attribute": ["value1", "value2"] }`
      RCTOKEN: Propagate attributes in the RCToken of the form:
        `"additional_claims": { "my_attribute": ["value1", "value2"] }`
    """
        OUTPUT_CREDENTIALS_UNSPECIFIED = 0
        HEADER = 1
        JWT = 2
        RCTOKEN = 3
    enable = _messages.BooleanField(1)
    expression = _messages.StringField(2)
    outputCredentials = _messages.EnumField('OutputCredentialsValueListEntryValuesEnum', 3, repeated=True)