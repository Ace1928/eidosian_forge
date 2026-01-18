from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FhirStore(_messages.Message):
    """Represents a FHIR store.

  Enums:
    VersionValueValuesEnum: Required. Immutable. The FHIR specification
      version that this FHIR store supports natively. This field is immutable
      after store creation. Requests are rejected if they contain FHIR
      resources of a different version. Version is required for every FHIR
      store.

  Messages:
    LabelsValue: User-supplied key-value pairs used to organize FHIR stores.
      Label keys must be between 1 and 63 characters long, have a UTF-8
      encoding of maximum 128 bytes, and must conform to the following PCRE
      regular expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must
      be between 1 and 63 characters long, have a UTF-8 encoding of maximum
      128 bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.

  Fields:
    disableReferentialIntegrity: Immutable. Whether to disable referential
      integrity in this FHIR store. This field is immutable after FHIR store
      creation. The default value is false, meaning that the API enforces
      referential integrity and fail the requests that result in inconsistent
      state in the FHIR store. When this field is set to true, the API skips
      referential integrity checks. Consequently, operations that rely on
      references, such as Patient-everything, do not return all the results if
      broken references exist.
    disableResourceVersioning: Immutable. Whether to disable resource
      versioning for this FHIR store. This field can not be changed after the
      creation of FHIR store. If set to false, which is the default behavior,
      all write operations cause historical versions to be recorded
      automatically. The historical versions can be fetched through the
      history APIs, but cannot be updated. If set to true, no historical
      versions are kept. The server sends errors for attempts to read the
      historical versions.
    enableUpdateCreate: Whether this FHIR store has the [updateCreate
      capability](https://www.hl7.org/fhir/capabilitystatement-
      definitions.html#CapabilityStatement.rest.resource.updateCreate). This
      determines if the client can use an Update operation to create a new
      resource with a client-specified ID. If false, all IDs are server-
      assigned through the Create operation and attempts to update a non-
      existent resource return errors. It is strongly advised not to include
      or encode any sensitive data such as patient identifiers in client-
      specified resource IDs. Those IDs are part of the FHIR resource path
      recorded in Cloud audit logs and Pub/Sub notifications. Those IDs can
      also be contained in reference fields within other resources.
    labels: User-supplied key-value pairs used to organize FHIR stores. Label
      keys must be between 1 and 63 characters long, have a UTF-8 encoding of
      maximum 128 bytes, and must conform to the following PCRE regular
      expression: \\p{Ll}\\p{Lo}{0,62} Label values are optional, must be
      between 1 and 63 characters long, have a UTF-8 encoding of maximum 128
      bytes, and must conform to the following PCRE regular expression:
      [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated
      with a given store.
    name: Output only. Identifier. Resource name of the FHIR store, of the
      form `projects/{project_id}/locations/{location}/datasets/{dataset_id}/f
      hirStores/{fhir_store_id}`.
    notificationConfig: Deprecated. Use `notification_configs` instead. If
      non-empty, publish all resource modifications of this FHIR store to this
      destination. The Pub/Sub message attributes contain a map with a string
      describing the action that has triggered the notification. For example,
      "action":"CreateResource".
    streamConfigs: A list of streaming configs that configure the destinations
      of streaming export for every resource mutation in this FHIR store. Each
      store is allowed to have up to 10 streaming configs. After a new config
      is added, the next resource mutation is streamed to the new location in
      addition to the existing ones. When a location is removed from the list,
      the server stops streaming to that location. Some lag (typically on the
      order of dozens of seconds) is expected before the results show up in
      the streaming destination.
    version: Required. Immutable. The FHIR specification version that this
      FHIR store supports natively. This field is immutable after store
      creation. Requests are rejected if they contain FHIR resources of a
      different version. Version is required for every FHIR store.
  """

    class VersionValueValuesEnum(_messages.Enum):
        """Required. Immutable. The FHIR specification version that this FHIR
    store supports natively. This field is immutable after store creation.
    Requests are rejected if they contain FHIR resources of a different
    version. Version is required for every FHIR store.

    Values:
      VERSION_UNSPECIFIED: VERSION_UNSPECIFIED is treated as STU3 to
        accommodate the existing FHIR stores.
      DSTU2: Draft Standard for Trial Use, [Release
        2](https://www.hl7.org/fhir/DSTU2)
      STU3: Standard for Trial Use, [Release 3](https://www.hl7.org/fhir/STU3)
      R4: [Release 4](https://www.hl7.org/fhir/R4)
    """
        VERSION_UNSPECIFIED = 0
        DSTU2 = 1
        STU3 = 2
        R4 = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """User-supplied key-value pairs used to organize FHIR stores. Label keys
    must be between 1 and 63 characters long, have a UTF-8 encoding of maximum
    128 bytes, and must conform to the following PCRE regular expression:
    \\p{Ll}\\p{Lo}{0,62} Label values are optional, must be between 1 and 63
    characters long, have a UTF-8 encoding of maximum 128 bytes, and must
    conform to the following PCRE regular expression:
    [\\p{Ll}\\p{Lo}\\p{N}_-]{0,63} No more than 64 labels can be associated with
    a given store.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    disableReferentialIntegrity = _messages.BooleanField(1)
    disableResourceVersioning = _messages.BooleanField(2)
    enableUpdateCreate = _messages.BooleanField(3)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    notificationConfig = _messages.MessageField('NotificationConfig', 6)
    streamConfigs = _messages.MessageField('StreamConfig', 7, repeated=True)
    version = _messages.EnumField('VersionValueValuesEnum', 8)