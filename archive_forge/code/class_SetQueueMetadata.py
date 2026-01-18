import json
import os
from osc_lib.command import command
from osc_lib import utils
from oslo_log import log as logging
from zaqarclient._i18n import _
from zaqarclient.queues.v1 import cli
class SetQueueMetadata(command.Command):
    """Set queue metadata"""
    _description = _('Set queue metadata')
    log = logging.getLogger(__name__ + '.SetQueueMetadata')

    def get_parser(self, prog_name):
        parser = super(SetQueueMetadata, self).get_parser(prog_name)
        parser.add_argument('queue_name', metavar='<queue_name>', help='Name of the queue')
        parser.add_argument('queue_metadata', metavar='<queue_metadata>', help='Queue metadata, All the metadata of the queue will be replaced by queue_metadata')
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        queue_name = parsed_args.queue_name
        queue_metadata = parsed_args.queue_metadata
        if client.api_version == 1 and (not client.queue(queue_name, auto_create=False).exists()):
            raise RuntimeError('Queue(%s) does not exist.' % queue_name)
        try:
            valid_metadata = json.loads(queue_metadata)
        except ValueError:
            raise RuntimeError('Queue metadata(%s) is not a valid json.' % queue_metadata)
        client.queue(queue_name, auto_create=False).metadata(new_meta=valid_metadata)