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
class GpgErrSig(GpgBaseError):
    """"It was not possible to check the signature.  This may be caused by
    a missing public key or an unsupported algorithm.  A RC of 4
    indicates unknown algorithm, a 9 indicates a missing public
    key.
    """
    keyid: str
    pkalgo: int
    hashalgo: int
    sig_class: str
    time: int
    rc: int
    fpr: str