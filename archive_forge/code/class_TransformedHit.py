import logging
import shutil
import sys
import textwrap
import xmlrpc.client
from collections import OrderedDict
from optparse import Values
from typing import TYPE_CHECKING, Dict, List, Optional
from pip._vendor.packaging.version import parse as parse_version
from pip._internal.cli.base_command import Command
from pip._internal.cli.req_command import SessionCommandMixin
from pip._internal.cli.status_codes import NO_MATCHES_FOUND, SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.metadata import get_default_environment
from pip._internal.models.index import PyPI
from pip._internal.network.xmlrpc import PipXmlrpcTransport
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import write_output
class TransformedHit(TypedDict):
    name: str
    summary: str
    versions: List[str]