from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
from googlecloudsdk.api_lib.app import logs_util
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.app import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.docker import constants
from googlecloudsdk.core.docker import docker
from googlecloudsdk.third_party.appengine.api import appinfo
def AddSslCertificateFlags(parser, required):
    """Add the common flags to an ssl-certificates command."""
    parser.add_argument('--display-name', required=required, help='A display name for this certificate.')
    parser.add_argument('--certificate', required=required, metavar='LOCAL_FILE_PATH', help='      The file path for the new certificate to upload. Must be in PEM\n      x.509 format including the header and footer.\n      ')
    parser.add_argument('--private-key', required=required, metavar='LOCAL_FILE_PATH', help='      The file path to a local RSA private key file. The private key must be\n      PEM encoded with header and footer and must be 2048 bits\n      or fewer.\n        ')