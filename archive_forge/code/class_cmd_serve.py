import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_serve(Command):
    __doc__ = 'Run the brz server.'
    aliases = ['server']
    takes_options = [Option('inet', help='Serve on stdin/out for use from inetd or sshd.'), RegistryOption('protocol', help='Protocol to serve.', lazy_registry=('breezy.transport', 'transport_server_registry'), value_switches=True), Option('listen', help='Listen for connections on nominated address.', type=str), Option('port', help='Listen for connections on nominated port.  Passing 0 as the port number will result in a dynamically allocated port.  The default port depends on the protocol.', type=int), custom_help('directory', help='Serve contents of this directory.'), Option('allow-writes', help='By default the server is a readonly server.  Supplying --allow-writes enables write access to the contents of the served directory and below.  Note that ``brz serve`` does not perform authentication, so unless some form of external authentication is arranged supplying this option leads to global uncontrolled write access to your file system.'), Option('client-timeout', type=float, help='Override the default idle client timeout (5min).')]

    def run(self, listen=None, port=None, inet=False, directory=None, allow_writes=False, protocol=None, client_timeout=None):
        from . import location, transport
        if directory is None:
            directory = osutils.getcwd()
        if protocol is None:
            protocol = transport.transport_server_registry.get()
        url = location.location_to_url(directory)
        if not allow_writes:
            url = 'readonly+' + url
        t = transport.get_transport_from_url(url)
        protocol(t, listen, port, inet, client_timeout)