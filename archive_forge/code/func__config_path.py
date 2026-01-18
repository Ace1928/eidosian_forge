import configparser
import os
import sys
import logging
import typing
from . import backend, credentials
from .util import platform_ as platform
from .backends import fail
def _config_path():
    return platform.config_root() / 'keyringrc.cfg'