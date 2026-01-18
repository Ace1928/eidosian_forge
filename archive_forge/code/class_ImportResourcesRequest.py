from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportResourcesRequest(_messages.Message):
    """Request to import resources.

  Enums:
    ContentStructureValueValuesEnum: The content structure in the source
      location. If not specified, the server treats the input source files as
      BUNDLE.

  Fields:
    contentStructure: The content structure in the source location. If not
      specified, the server treats the input source files as BUNDLE.
    gcsErrorDestination: The Cloud Storage destination to write the error
      report to. The Healthcare Service Agent account requires the
      `roles/storage.objectAdmin` role on the Cloud Storage location. Writing
      a file to the same destination multiple times results in the previous
      version of the file being overwritten.
    gcsSource: Cloud Storage source data location and import configuration.
      The Healthcare Service Agent account requires the
      `roles/storage.objectAdmin` role on the Cloud Storage location. For each
      Cloud Storage object, use a text file that contains the format specified
      in ContentStructure.
  """

    class ContentStructureValueValuesEnum(_messages.Enum):
        """The content structure in the source location. If not specified, the
    server treats the input source files as BUNDLE.

    Values:
      CONTENT_STRUCTURE_UNSPECIFIED: If the content structure is not
        specified, the default value `BUNDLE` is used.
      BUNDLE: The source file contains one or more lines of newline-delimited
        JSON (ndjson). Each line is a bundle that contains one or more
        resources.
      RESOURCE: The source file contains one or more lines of newline-
        delimited JSON (ndjson). Each line is a single resource.
      BUNDLE_PRETTY: The entire file is one JSON bundle. The JSON can span
        multiple lines.
      RESOURCE_PRETTY: The entire file is one JSON resource. The JSON can span
        multiple lines.
    """
        CONTENT_STRUCTURE_UNSPECIFIED = 0
        BUNDLE = 1
        RESOURCE = 2
        BUNDLE_PRETTY = 3
        RESOURCE_PRETTY = 4
    contentStructure = _messages.EnumField('ContentStructureValueValuesEnum', 1)
    gcsErrorDestination = _messages.MessageField('GoogleCloudHealthcareV1alpha2FhirGcsErrorDestination', 2)
    gcsSource = _messages.MessageField('GoogleCloudHealthcareV1alpha2FhirGcsSource', 3)