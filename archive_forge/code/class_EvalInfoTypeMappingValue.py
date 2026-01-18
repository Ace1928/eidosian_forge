from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class EvalInfoTypeMappingValue(_messages.Message):
    """Optional. InfoType mapping for `eval_store`. Different resources can
    map to the same infoType. For example, `PERSON_NAME`, `PERSON`, `NAME`,
    and `HUMAN` are different. To map all of these into a single infoType
    (such as `PERSON_NAME`), specify the following mapping: ```
    info_type_mapping["PERSON"] = "PERSON_NAME" info_type_mapping["NAME"] =
    "PERSON_NAME" info_type_mapping["HUMAN"] = "PERSON_NAME" ``` Unmentioned
    infoTypes, such as `DATE`, are treated as identity mapping. For example:
    ``` info_type_mapping["DATE"] = "DATE" ``` InfoTypes are case-insensitive.

    Messages:
      AdditionalProperty: An additional property for a
        EvalInfoTypeMappingValue object.

    Fields:
      additionalProperties: Additional properties of type
        EvalInfoTypeMappingValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a EvalInfoTypeMappingValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)