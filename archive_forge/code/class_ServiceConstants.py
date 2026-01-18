from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ServiceConstants(proto.Message):
    """Shared constants.
    """

    class Values(proto.Enum):
        """A collection of constant values meaningful to the Storage
        API.

        Values:
            VALUES_UNSPECIFIED (0):
                Unused. Proto3 requires first enum to be 0.
            MAX_READ_CHUNK_BYTES (2097152):
                The maximum size chunk that can will be
                returned in a single ReadRequest.
                2 MiB.
            MAX_WRITE_CHUNK_BYTES (2097152):
                The maximum size chunk that can be sent in a
                single WriteObjectRequest. 2 MiB.
            MAX_OBJECT_SIZE_MB (5242880):
                The maximum size of an object in MB - whether
                written in a single stream or composed from
                multiple other objects. 5 TiB.
            MAX_CUSTOM_METADATA_FIELD_NAME_BYTES (1024):
                The maximum length field name that can be
                sent in a single custom metadata field.
                1 KiB.
            MAX_CUSTOM_METADATA_FIELD_VALUE_BYTES (4096):
                The maximum length field value that can be sent in a single
                custom_metadata field. 4 KiB.
            MAX_CUSTOM_METADATA_TOTAL_SIZE_BYTES (8192):
                The maximum total bytes that can be populated into all field
                names and values of the custom_metadata for one object. 8
                KiB.
            MAX_BUCKET_METADATA_TOTAL_SIZE_BYTES (20480):
                The maximum total bytes that can be populated
                into all bucket metadata fields.
                20 KiB.
            MAX_NOTIFICATION_CONFIGS_PER_BUCKET (100):
                The maximum number of NotificationConfigs
                that can be registered for a given bucket.
            MAX_LIFECYCLE_RULES_PER_BUCKET (100):
                The maximum number of LifecycleRules that can
                be registered for a given bucket.
            MAX_NOTIFICATION_CUSTOM_ATTRIBUTES (5):
                The maximum number of custom attributes per
                NotificationConfigs.
            MAX_NOTIFICATION_CUSTOM_ATTRIBUTE_KEY_LENGTH (256):
                The maximum length of a custom attribute key
                included in NotificationConfig.
            MAX_NOTIFICATION_CUSTOM_ATTRIBUTE_VALUE_LENGTH (1024):
                The maximum length of a custom attribute
                value included in a NotificationConfig.
            MAX_LABELS_ENTRIES_COUNT (64):
                The maximum number of key/value entries per
                bucket label.
            MAX_LABELS_KEY_VALUE_LENGTH (63):
                The maximum character length of the key or
                value in a bucket label map.
            MAX_LABELS_KEY_VALUE_BYTES (128):
                The maximum byte size of the key or value in
                a bucket label map.
            MAX_OBJECT_IDS_PER_DELETE_OBJECTS_REQUEST (1000):
                The maximum number of object IDs that can be
                included in a DeleteObjectsRequest.
            SPLIT_TOKEN_MAX_VALID_DAYS (14):
                The maximum number of days for which a token
                returned by the GetListObjectsSplitPoints RPC is
                valid.
        """
        _pb_options = {'allow_alias': True}
        VALUES_UNSPECIFIED = 0
        MAX_READ_CHUNK_BYTES = 2097152
        MAX_WRITE_CHUNK_BYTES = 2097152
        MAX_OBJECT_SIZE_MB = 5242880
        MAX_CUSTOM_METADATA_FIELD_NAME_BYTES = 1024
        MAX_CUSTOM_METADATA_FIELD_VALUE_BYTES = 4096
        MAX_CUSTOM_METADATA_TOTAL_SIZE_BYTES = 8192
        MAX_BUCKET_METADATA_TOTAL_SIZE_BYTES = 20480
        MAX_NOTIFICATION_CONFIGS_PER_BUCKET = 100
        MAX_LIFECYCLE_RULES_PER_BUCKET = 100
        MAX_NOTIFICATION_CUSTOM_ATTRIBUTES = 5
        MAX_NOTIFICATION_CUSTOM_ATTRIBUTE_KEY_LENGTH = 256
        MAX_NOTIFICATION_CUSTOM_ATTRIBUTE_VALUE_LENGTH = 1024
        MAX_LABELS_ENTRIES_COUNT = 64
        MAX_LABELS_KEY_VALUE_LENGTH = 63
        MAX_LABELS_KEY_VALUE_BYTES = 128
        MAX_OBJECT_IDS_PER_DELETE_OBJECTS_REQUEST = 1000
        SPLIT_TOKEN_MAX_VALID_DAYS = 14