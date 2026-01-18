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
class AddFloatingIP(command.Command):
    """Add floating IP address to container"""
    log = logging.getLogger(__name__ + '.AddFloatingIP')

    def get_parser(self, prog_name):
        parser = super(AddFloatingIP, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help='ID or name of the container to receive the floating IP address.')
        parser.add_argument('ip_address', metavar='<ip-address>', help='Floating IP address to assign to the first available container port (IP only)')
        parser.add_argument('--fixed-ip-address', metavar='<ip-address>', help='Fixed IP address to associate with this floating IP address. The first container port containing the fixed IP address will be used')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        opts = {}
        opts['id'] = parsed_args.container
        container = client.containers.get(**opts)
        network_client = self.app.client_manager.network
        attrs = {}
        obj = network_client.find_ip(parsed_args.ip_address, ignore_missing=False)
        ports = list(network_client.ports(device_id=container.uuid))
        if parsed_args.fixed_ip_address:
            fip_address = parsed_args.fixed_ip_address
            attrs['fixed_ip_address'] = fip_address
            for port in ports:
                for ip in port.fixed_ips:
                    if ip['ip_address'] == fip_address:
                        attrs['port_id'] = port.id
                        break
                else:
                    continue
                break
            if 'port_id' not in attrs:
                print(_('No port found for fixed IP address %s.') % fip_address)
                raise SystemExit
            network_client.update_ip(obj, **attrs)
        else:
            error = None
            for port in ports:
                attrs['port_id'] = port.id
                try:
                    network_client.update_ip(obj, **attrs)
                except Exception as e:
                    error = e
                    continue
                else:
                    error = None
                    break
            if error:
                raise error