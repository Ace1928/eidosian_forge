import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def _load_keyring_path(config: configparser.RawConfigParser) -> None:
    """load the keyring-path option (if present)"""
    try:
        path = config.get('backend', 'keyring-path').strip()
        sys.path.insert(0, path)
    except (configparser.NoOptionError, configparser.NoSectionError):
        pass