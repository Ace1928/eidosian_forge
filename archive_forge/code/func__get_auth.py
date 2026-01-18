from __future__ import annotations
import os
from typing import (
@staticmethod
def _get_auth() -> Union[tuple, None]:
    """
        Returns the basic authentication configuration
        """
    username = os.environ.get('GRAPHDB_USERNAME', None)
    password = os.environ.get('GRAPHDB_PASSWORD', None)
    if username:
        if not password:
            raise ValueError("Environment variable 'GRAPHDB_USERNAME' is set, but 'GRAPHDB_PASSWORD' is not set.")
        else:
            return (username, password)
    return None