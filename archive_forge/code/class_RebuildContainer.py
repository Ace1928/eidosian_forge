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
class RebuildContainer(command.Command):
    """Rebuild one or more running container(s)"""
    log = logging.getLogger(__name__ + '.RebuildContainer')

    def get_parser(self, prog_name):
        parser = super(RebuildContainer, self).get_parser(prog_name)
        parser.add_argument('containers', metavar='<container>', nargs='+', help='ID or name of the (container)s to rebuild.')
        parser.add_argument('--image', metavar='<image>', help='The image for specified container to update.')
        parser.add_argument('--image-driver', metavar='<image_driver>', help='The image driver to use to update container image. It can have following values: "docker": update the image from Docker Hub. "glance": update the image from Glance. The default value is source container\'s image driver ')
        parser.add_argument('--wait', action='store_true', help='Wait for rebuild to complete')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        for container in parsed_args.containers:
            opts = {}
            opts['id'] = container
            if parsed_args.image:
                opts['image'] = parsed_args.image
            if parsed_args.image_driver:
                opts['image_driver'] = parsed_args.image_driver
            try:
                client.containers.rebuild(**opts)
                print(_('Request to rebuild container %s has been accepted') % container)
                if parsed_args.wait:
                    if utils.wait_for_status(client.containers.get, container, success_status=['created', 'running']):
                        print('rebuild container %(container)s success.' % {'container': container})
                    else:
                        print('rebuild container %(container)s failed.' % {'container': container})
                        raise SystemExit
            except Exception as e:
                print('rebuild container %(container)s failed: %(e)s' % {'container': container, 'e': e})