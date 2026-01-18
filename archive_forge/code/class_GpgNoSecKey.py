from ansible.errors import AnsibleError
from ansible.galaxy.user_agent import user_agent
from ansible.module_utils.urls import open_url
import contextlib
import os
import subprocess
import sys
import typing as t
from dataclasses import dataclass, fields as dc_fields
from functools import partial
from urllib.error import HTTPError, URLError
@frozen_dataclass
class GpgNoSecKey(GpgBaseError):
    """The secret key is not available."""
    keyid: str