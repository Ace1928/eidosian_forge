from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import (
from cryptography.hazmat.primitives import hashes
class CRLEntryExtensionOID:
    CERTIFICATE_ISSUER = ObjectIdentifier('2.5.29.29')
    CRL_REASON = ObjectIdentifier('2.5.29.21')
    INVALIDITY_DATE = ObjectIdentifier('2.5.29.24')