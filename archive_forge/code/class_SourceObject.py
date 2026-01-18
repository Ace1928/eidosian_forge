from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class SourceObject(proto.Message):
    """Description of a source object for a composition request.

        Attributes:
            name (str):
                Required. The source object's name. All
                source objects must reside in the same bucket.
            generation (int):
                The generation of this object to use as the
                source.
            object_preconditions (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ComposeObjectRequest.SourceObject.ObjectPreconditions):
                Conditions that must be met for this
                operation to execute.
        """

    class ObjectPreconditions(proto.Message):
        """Preconditions for a source object of a composition request.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                if_generation_match (int):
                    Only perform the composition if the
                    generation of the source object that would be
                    used matches this value.  If this value and a
                    generation are both specified, they must be the
                    same value or the call will fail.

                    This field is a member of `oneof`_ ``_if_generation_match``.
            """
        if_generation_match: int = proto.Field(proto.INT64, number=1, optional=True)
    name: str = proto.Field(proto.STRING, number=1)
    generation: int = proto.Field(proto.INT64, number=2)
    object_preconditions: 'ComposeObjectRequest.SourceObject.ObjectPreconditions' = proto.Field(proto.MESSAGE, number=3, message='ComposeObjectRequest.SourceObject.ObjectPreconditions')