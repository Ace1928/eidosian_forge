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
class SearchCommand(Command, SessionCommandMixin):
    """Search for PyPI packages whose name or summary contains <query>."""
    usage = '\n      %prog [options] <query>'
    ignore_require_venv = True

    def add_options(self) -> None:
        self.cmd_opts.add_option('-i', '--index', dest='index', metavar='URL', default=PyPI.pypi_url, help='Base URL of Python Package Index (default %default)')
        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        if not args:
            raise CommandError('Missing required argument (search query).')
        query = args
        pypi_hits = self.search(query, options)
        hits = transform_hits(pypi_hits)
        terminal_width = None
        if sys.stdout.isatty():
            terminal_width = shutil.get_terminal_size()[0]
        print_results(hits, terminal_width=terminal_width)
        if pypi_hits:
            return SUCCESS
        return NO_MATCHES_FOUND

    def search(self, query: List[str], options: Values) -> List[Dict[str, str]]:
        index_url = options.index
        session = self.get_default_session(options)
        transport = PipXmlrpcTransport(index_url, session)
        pypi = xmlrpc.client.ServerProxy(index_url, transport)
        try:
            hits = pypi.search({'name': query, 'summary': query}, 'or')
        except xmlrpc.client.Fault as fault:
            message = 'XMLRPC request failed [code: {code}]\n{string}'.format(code=fault.faultCode, string=fault.faultString)
            raise CommandError(message)
        assert isinstance(hits, list)
        return hits