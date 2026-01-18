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
def _find_alarm_id_by_name(client, name):
    alarm = _find_alarm_by_name(client, name)
    return alarm['alarm_id']