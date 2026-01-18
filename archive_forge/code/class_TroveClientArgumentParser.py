import argparse
import getpass
import glob
import importlib
import itertools
import logging
import os
import pkgutil
import sys
from keystoneauth1.identity.generic import password
from keystoneauth1.identity.generic import token
from keystoneauth1 import loading
from oslo_utils import encodeutils
from oslo_utils import importutils
import pkg_resources
from troveclient.apiclient import exceptions as exc
import troveclient.auth_plugin
from troveclient import client
import troveclient.extension
from troveclient.i18n import _  # noqa
from troveclient import utils
from troveclient.v1 import shell as shell_v1
class TroveClientArgumentParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super(TroveClientArgumentParser, self).__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        if kwargs.get('help') is None:
            raise Exception(_("An argument '%s' was specified without help.") % args[0])
        super(TroveClientArgumentParser, self).add_argument(*args, **kwargs)

    def error(self, message):
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.
        """
        self.print_usage(sys.stderr)
        choose_from = ' (choose from'
        progparts = self.prog.partition(' ')
        self.exit(2, "error: %(errmsg)s\nTry '%(mainp)s help %(subp)s' for more information.\n" % {'errmsg': message.split(choose_from)[0], 'mainp': progparts[0], 'subp': progparts[2]})