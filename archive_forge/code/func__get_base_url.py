import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@staticmethod
def _get_base_url(url: Optional[str]=None) -> str:
    """Return the base URL with the trailing slash stripped.
        If the URL is a Falsy value, return the default URL.
        Returns:
            The base URL
        """
    if not url:
        return gitlab.const.DEFAULT_URL
    return url.rstrip('/')