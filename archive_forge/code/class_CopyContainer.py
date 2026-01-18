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
class CopyContainer(command.Command):
    """Copy files/tars between a container and the local filesystem."""
    log = logging.getLogger(__name__ + '.CopyContainer')

    def get_parser(self, prog_name):
        parser = super(CopyContainer, self).get_parser(prog_name)
        parser.add_argument('source', metavar='<source>', help='The source should be copied to the container or localhost. The format of this parameter is [container:]src_path.')
        parser.add_argument('destination', metavar='<destination>', help='The directory destination where save the source. The format of this parameter is [container:]dest_path.')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        if ':' in parsed_args.source:
            source_parts = parsed_args.source.split(':', 1)
            container_id = source_parts[0]
            container_path = source_parts[1]
            opts = {}
            opts['id'] = container_id
            opts['path'] = container_path
            res = client.containers.get_archive(**opts)
            dest_path = parsed_args.destination
            tardata = io.BytesIO(res['data'])
            with closing(tarfile.open(fileobj=tardata)) as tar:
                tar.extractall(dest_path)
        elif ':' in parsed_args.destination:
            dest_parts = parsed_args.destination.split(':', 1)
            container_id = dest_parts[0]
            container_path = dest_parts[1]
            filename = os.path.split(parsed_args.source)[1]
            opts = {}
            opts['id'] = container_id
            opts['path'] = container_path
            tardata = io.BytesIO()
            with closing(tarfile.open(fileobj=tardata, mode='w')) as tar:
                tar.add(parsed_args.source, arcname=filename)
            opts['data'] = tardata.getvalue()
            client.containers.put_archive(**opts)
        else:
            print('Please check the parameters for zun copy!')
            print('Usage:')
            print('openstack appcontainer cp container:src_path dest_path|-')
            print('openstack appcontainer cp src_path|- container:dest_path')