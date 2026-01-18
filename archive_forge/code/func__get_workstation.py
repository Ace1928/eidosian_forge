import base64
import logging
import os
import socket
import typing
from spnego._context import (
from spnego._credential import (
from spnego._ntlm_raw.crypto import (
from spnego._ntlm_raw.messages import (
from spnego._ntlm_raw.security import seal, sign
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
from spnego.iov import BufferType, IOVResBuffer
def _get_workstation() -> typing.Optional[str]:
    """Get the current workstation name.

    This gets the current workstation name that respects `NETBIOS_COMPUTER_NAME`. The env var is used by the library
    that gss-ntlmssp calls and makes sure that this Python implementation is a closer in its behaviour.

    Returns:
        Optional[str]: The workstation to supply in the NTLM authentication message or None.
    """
    if 'NETBIOS_COMPUTER_NAME' in os.environ:
        workstation = os.environ['NETBIOS_COMPUTER_NAME']
    else:
        workstation = socket.gethostname().upper()
    return to_text(workstation) if workstation else None