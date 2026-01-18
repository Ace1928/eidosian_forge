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
class GpgNoData(GpgBaseError):
    """No data has been found.  Codes for WHAT are:
    - 1 :: No armored data.
    - 2 :: Expected a packet but did not find one.
    - 3 :: Invalid packet found, this may indicate a non OpenPGP
           message.
    - 4 :: Signature expected but not found.
    """
    what: str