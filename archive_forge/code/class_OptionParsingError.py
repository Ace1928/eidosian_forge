import logging
import optparse
import os
import re
import shlex
import urllib.parse
from optparse import Values
from typing import (
from pip._internal.cli import cmdoptions
from pip._internal.exceptions import InstallationError, RequirementsFileParseError
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.encoding import auto_decode
from pip._internal.utils.urls import get_url_scheme
class OptionParsingError(Exception):

    def __init__(self, msg: str) -> None:
        self.msg = msg