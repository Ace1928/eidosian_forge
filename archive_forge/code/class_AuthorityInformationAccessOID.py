from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import (
from cryptography.hazmat.primitives import hashes
class AuthorityInformationAccessOID:
    CA_ISSUERS = ObjectIdentifier('1.3.6.1.5.5.7.48.2')
    OCSP = ObjectIdentifier('1.3.6.1.5.5.7.48.1')