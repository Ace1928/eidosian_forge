from __future__ import annotations
import base64
import collections.abc
import logging
import os
import typing as t
from spnego._context import (
from spnego._credential import Credential, CredentialCache, Password, unify_credentials
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.exceptions import WinError as NativeError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
def _available_protocols() -> list[str]:
    """Return a list of protocols that SSPIProxy can offer."""
    if HAS_SSPI:
        return ['kerberos', 'negotiate', 'ntlm']
    else:
        return []