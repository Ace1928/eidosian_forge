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
def _check_name_and_id(parsed_args, action):
    _check_name_and_id_coexist(parsed_args, action)
    _check_name_and_id_exist(parsed_args, action)