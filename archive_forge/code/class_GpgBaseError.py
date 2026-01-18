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
class GpgBaseError(Exception):
    status: str

    @classmethod
    def get_gpg_error_description(cls) -> str:
        """Return the current class description."""
        return ' '.join(cls.__doc__.split())

    def __post_init__(self):
        for field in dc_fields(self):
            super(GpgBaseError, self).__setattr__(field.name, field.type(getattr(self, field.name)))