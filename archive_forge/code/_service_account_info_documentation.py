import io
import json
import six
from google.auth import crypt
from google.auth import exceptions
Reads a Google service account JSON file and returns its parsed info.

    Args:
        filename (str): The path to the service account .json file.
        require (Sequence[str]): List of keys required to be present in the
            info.
        use_rsa_signer (Optional[bool]): Whether to use RSA signer or EC signer.
            We use RSA signer by default.

    Returns:
        Tuple[ Mapping[str, str], google.auth.crypt.Signer ]: The verified
            info and a signer instance.
    