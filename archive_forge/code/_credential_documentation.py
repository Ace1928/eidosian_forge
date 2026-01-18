import dataclasses
import typing
from spnego._ntlm_raw.crypto import is_ntlm_hash
from spnego.exceptions import InvalidCredentialError, NoCredentialError
List of protocols the credential can be used for.