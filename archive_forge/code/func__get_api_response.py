import os
import sys
import ftplib
import warnings
from .utils import parse_url
@classmethod
def _get_api_response(cls, doi, archive_url):
    """
        Perform the actual API request

        This has been separated into a separate ``classmethod``, as it can be
        used prior and after the initialization.
        """
    import requests
    parsed = parse_url(archive_url)
    response = requests.get(f'{parsed['protocol']}://{parsed['netloc']}/api/datasets/:persistentId?persistentId=doi:{doi}', timeout=5)
    return response