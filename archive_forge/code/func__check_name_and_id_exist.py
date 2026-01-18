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
def _check_name_and_id_exist(parsed_args, action):
    if not parsed_args.id and (not parsed_args.name):
        msg = _('You need to specify one of alarm ID and alarm name(--name) to %s an alarm.') % action
        raise exceptions.CommandError(msg)