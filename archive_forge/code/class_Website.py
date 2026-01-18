from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class Website(proto.Message):
    """Properties of a bucket related to accessing the contents as a
        static website. For more on hosting a static website via Cloud
        Storage, see
        https://cloud.google.com/storage/docs/hosting-static-website.

        Attributes:
            main_page_suffix (str):
                If the requested object path is missing, the service will
                ensure the path has a trailing '/', append this suffix, and
                attempt to retrieve the resulting object. This allows the
                creation of ``index.html`` objects to represent directory
                pages.
            not_found_page (str):
                If the requested object path is missing, and any
                ``mainPageSuffix`` object is missing, if applicable, the
                service will return the named object from this bucket as the
                content for a
                [https://tools.ietf.org/html/rfc7231#section-6.5.4][404 Not
                Found] result.
        """
    main_page_suffix: str = proto.Field(proto.STRING, number=1)
    not_found_page: str = proto.Field(proto.STRING, number=2)