import json
import logging
from os import path
import re
import subprocess
import six
from google.auth import exceptions
def _check_dca_metadata_path(metadata_path):
    """Checks for context aware metadata. If it exists, returns the absolute path;
    otherwise returns None.

    Args:
        metadata_path (str): context aware metadata path.

    Returns:
        str: absolute path if exists and None otherwise.
    """
    metadata_path = path.expanduser(metadata_path)
    if not path.exists(metadata_path):
        _LOGGER.debug('%s is not found, skip client SSL authentication.', metadata_path)
        return None
    return metadata_path