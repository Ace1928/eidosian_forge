import abc
import dataclasses
import enum
import typing
import warnings
from spnego._credential import Credential
from spnego._text import to_text
from spnego.channel_bindings import GssChannelBindings
from spnego.exceptions import FeatureMissingError, NegotiateOptions, SpnegoError
from spnego.iov import BufferType, IOVBuffer, IOVResBuffer
Reset the NTLM crypto handles after signing/verifying the SPNEGO mechListMIC.

        `MS-SPNG`_ documents that after signing or verifying the mechListMIC, the RC4 key state needs to be the same
        for the mechListMIC and for the first message signed/sealed by the application. Because we use SSPI on Windows
        hosts which does all the work for us this function only matters for Linux hosts.

        Args:
            outgoing: Whether to reset the outgoing or incoming RC4 key state.

        .. _MS-SPNG:
            https://docs.microsoft.com/en-us/openspecs/windows_protocols/ms-spng/b87587b3-9d72-4027-8131-b76b5368115f
        