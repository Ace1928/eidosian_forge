import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class OldCreateClaim(cli.OldCreateClaim):
    """Create claim and return a list of claimed messages"""
    pass