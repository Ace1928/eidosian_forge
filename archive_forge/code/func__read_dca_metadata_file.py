import json
import logging
from os import path
import re
import subprocess
import six
from google.auth import exceptions
def _read_dca_metadata_file(metadata_path):
    """Loads context aware metadata from the given path.

    Args:
        metadata_path (str): context aware metadata path.

    Returns:
        Dict[str, str]: The metadata.

    Raises:
        google.auth.exceptions.ClientCertError: If failed to parse metadata as JSON.
    """
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except ValueError as caught_exc:
        new_exc = exceptions.ClientCertError(caught_exc)
        six.raise_from(new_exc, caught_exc)
    return metadata