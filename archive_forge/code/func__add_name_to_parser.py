import argparse
from cliff import command
from cliff import lister
from cliff import show
from oslo_serialization import jsonutils
from oslo_utils import strutils
from oslo_utils import uuidutils
from aodhclient import exceptions
from aodhclient.i18n import _
from aodhclient import utils
def _add_name_to_parser(parser, required=False):
    parser.add_argument('--name', metavar='<NAME>', required=required, help='Name of the alarm')
    return parser