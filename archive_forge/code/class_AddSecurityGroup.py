import argparse
from contextlib import closing
import io
import os
from oslo_log import log as logging
import tarfile
import time
from osc_lib.command import command
from osc_lib import utils
from zunclient.common import utils as zun_utils
from zunclient.common.websocketclient import exceptions
from zunclient.common.websocketclient import websocketclient
from zunclient import exceptions as exc
from zunclient.i18n import _
class AddSecurityGroup(command.Command):
    """Add security group for specified container."""
    log = logging.getLogger(__name__ + '.AddSecurityGroup')

    def get_parser(self, prog_name):
        parser = super(AddSecurityGroup, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to add security group.')
        parser.add_argument('security_group', metavar='<security_group>', help='Security group ID or name for specified container. ')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.container
        opts['security_group'] = parsed_args.security_group
        opts = zun_utils.remove_null_parms(**opts)
        try:
            client.containers.add_security_group(**opts)
            print('Request to add security group for container %s has been accepted.' % parsed_args.container)
        except Exception as e:
            print('Add security group for container %(container)s failed: %(e)s' % {'container': parsed_args.container, 'e': e})