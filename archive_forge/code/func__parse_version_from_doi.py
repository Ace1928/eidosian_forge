import os
import sys
import ftplib
import warnings
from .utils import parse_url
def _parse_version_from_doi(self):
    """
        Parse version from the doi

        Return None if version is not available in the doi.
        """
    _, suffix = self.doi.split('/')
    last_part = suffix.split('.')[-1]
    if last_part[0] != 'v':
        return None
    version = int(last_part[1:])
    return version