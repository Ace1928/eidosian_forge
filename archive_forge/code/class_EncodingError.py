import boto.exception
from boto.compat import json
import requests
import boto
class EncodingError(Exception):
    """
    Content sent for Cloud Search indexing was incorrectly encoded.

    This usually happens when a document is marked as unicode but non-unicode
    characters are present.
    """
    pass