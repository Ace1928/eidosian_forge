import base64
import functools
import inspect
import json
import logging
import os
import warnings
import six
from six.moves import urllib
def _parse_pem_key(raw_key_input):
    """Identify and extract PEM keys.

    Determines whether the given key is in the format of PEM key, and extracts
    the relevant part of the key if it is.

    Args:
        raw_key_input: The contents of a private key file (either PEM or
                       PKCS12).

    Returns:
        string, The actual key if the contents are from a PEM file, or
        else None.
    """
    offset = raw_key_input.find(b'-----BEGIN ')
    if offset != -1:
        return raw_key_input[offset:]