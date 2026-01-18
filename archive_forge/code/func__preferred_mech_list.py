import base64
import logging
import struct
import typing
import spnego
from spnego._context import (
from spnego._credential import Credential, unify_credentials
from spnego._gss import GSSAPIProxy
from spnego._spnego import (
from spnego._sspi import SSPIProxy
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import (
def _preferred_mech_list(self) -> typing.List[GSSMech]:
    """Get a list of mechs that can be used in priority order (highest to lowest)."""
    available_protocols = [p for p in self.available_protocols(self.options) if p != 'negotiate']
    return [getattr(GSSMech, p) for p in available_protocols]