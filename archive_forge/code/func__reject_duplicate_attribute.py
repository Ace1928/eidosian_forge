from __future__ import annotations
import abc
import datetime
import os
import typing
from cryptography import utils
from cryptography.hazmat.bindings._rust import x509 as rust_x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import (
from cryptography.hazmat.primitives.asymmetric.types import (
from cryptography.x509.extensions import (
from cryptography.x509.name import Name, _ASN1Type
from cryptography.x509.oid import ObjectIdentifier
def _reject_duplicate_attribute(oid: ObjectIdentifier, attributes: typing.List[typing.Tuple[ObjectIdentifier, bytes, typing.Optional[int]]]) -> None:
    for attr_oid, _, _ in attributes:
        if attr_oid == oid:
            raise ValueError('This attribute has already been set.')