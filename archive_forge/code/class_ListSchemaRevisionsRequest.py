from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class ListSchemaRevisionsRequest(proto.Message):
    """Request for the ``ListSchemaRevisions`` method.

    Attributes:
        name (str):
            Required. The name of the schema to list
            revisions for.
        view (google.pubsub_v1.types.SchemaView):
            The set of Schema fields to return in the response. If not
            set, returns Schemas with ``name`` and ``type``, but not
            ``definition``. Set to ``FULL`` to retrieve all fields.
        page_size (int):
            The maximum number of revisions to return per
            page.
        page_token (str):
            The page token, received from a previous
            ListSchemaRevisions call. Provide this to
            retrieve the subsequent page.
    """
    name: str = proto.Field(proto.STRING, number=1)
    view: 'SchemaView' = proto.Field(proto.ENUM, number=2, enum='SchemaView')
    page_size: int = proto.Field(proto.INT32, number=3)
    page_token: str = proto.Field(proto.STRING, number=4)