from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JsonExtensionValueValuesEnum(_messages.Enum):
    """Optional. Load option to be used together with source_format newline-
    delimited JSON to indicate that a variant of JSON is being loaded. To load
    newline-delimited GeoJSON, specify GEOJSON (and source_format must be set
    to NEWLINE_DELIMITED_JSON).

    Values:
      JSON_EXTENSION_UNSPECIFIED: The default if provided value is not one
        included in the enum, or the value is not specified. The source
        formate is parsed without any modification.
      GEOJSON: Use GeoJSON variant of JSON. See
        https://tools.ietf.org/html/rfc7946.
    """
    JSON_EXTENSION_UNSPECIFIED = 0
    GEOJSON = 1