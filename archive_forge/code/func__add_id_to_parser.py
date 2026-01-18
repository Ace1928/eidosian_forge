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
def _add_id_to_parser(parser):
    parser.add_argument('id', nargs='?', metavar='<ALARM ID or NAME>', help='ID or name of an alarm.')
    return parser