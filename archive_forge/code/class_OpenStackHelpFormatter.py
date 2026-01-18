import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
class OpenStackHelpFormatter(argparse.HelpFormatter):

    def start_section(self, heading):
        heading = '%s%s' % (heading[0].upper(), heading[1:])
        super(OpenStackHelpFormatter, self).start_section(heading)