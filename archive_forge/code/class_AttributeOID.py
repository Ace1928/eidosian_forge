from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import (
from cryptography.hazmat.primitives import hashes
class AttributeOID:
    CHALLENGE_PASSWORD = ObjectIdentifier('1.2.840.113549.1.9.7')
    UNSTRUCTURED_NAME = ObjectIdentifier('1.2.840.113549.1.9.2')