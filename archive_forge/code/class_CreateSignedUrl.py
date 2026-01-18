import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class CreateSignedUrl(command.ShowOne):
    """Create a pre-signed url"""
    _description = _('Create a pre-signed url')
    log = logging.getLogger(__name__ + '.CreateSignedUrl')

    def get_parser(self, prog_name):
        parser = super(CreateSignedUrl, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue')
        parser.add_argument('--paths', metavar='<paths>', default='messages', help='Allowed paths in a comma-separated list. Options: messages, subscriptions, claims')
        parser.add_argument('--ttl-seconds', metavar='<ttl_seconds>', type=int, help='Length of time (in seconds) until the signature expires')
        parser.add_argument('--methods', metavar='<methods>', default='GET', help='HTTP methods to allow as a comma-separated list. Options: GET, HEAD, OPTIONS, POST, PUT, DELETE')
        return parser
    allowed_paths = ('messages', 'subscriptions', 'claims')

    def take_action(self, parsed_args):
        client = self.app.client_manager.messaging
        queue = client.queue(parsed_args.queue_name, auto_create=False)
        paths = parsed_args.paths.split(',')
        if not all([p in self.allowed_paths for p in paths]):
            print('Invalid path supplied! Received {}. Valid paths are: messages, subscriptions, claims'.format(','.join(paths)))
        kwargs = {'methods': parsed_args.methods.split(','), 'paths': paths}
        if parsed_args.ttl_seconds:
            kwargs['ttl_seconds'] = parsed_args.ttl_seconds
        data = queue.signed_url(**kwargs)
        fields = ('Paths', 'Methods', 'Expires', 'Signature', 'Project ID')
        return (fields, (','.join(data['paths']), ','.join(data['methods']), data['expires'], data['signature'], data['project']))