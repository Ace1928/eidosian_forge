import argparse
import datetime
import logging
import re
from oslo_serialization import jsonutils
from oslo_utils import strutils
from blazarclient import command
from blazarclient import exception
class ShowLease(command.ShowCommand):
    """Show details about the given lease."""
    resource = 'lease'
    json_indent = 4
    log = logging.getLogger(__name__ + '.ShowLease')

    def get_parser(self, prog_name):
        parser = super(ShowLease, self).get_parser(prog_name)
        if self.allow_names:
            help_str = 'ID or name of %s to look up'
        else:
            help_str = 'ID of %s to look up'
        parser.add_argument('id', metavar=self.resource.upper(), help=help_str % self.resource)
        return parser