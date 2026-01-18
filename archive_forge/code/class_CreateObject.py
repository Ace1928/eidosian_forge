import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class CreateObject(command.Lister):
    _description = _('Upload object to container')

    def get_parser(self, prog_name):
        parser = super(CreateObject, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Container for new object'))
        parser.add_argument('objects', metavar='<filename>', nargs='+', help=_('Local filename(s) to upload'))
        parser.add_argument('--name', metavar='<name>', help=_('Upload a file and rename it. Can only be used when uploading a single object'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.name:
            if len(parsed_args.objects) > 1:
                msg = _('Attempting to upload multiple objects and using --name is not permitted')
                raise exceptions.CommandError(msg)
        results = []
        for obj in parsed_args.objects:
            if len(obj) > 1024:
                LOG.warning(_('Object name is %s characters long, default limit is 1024'), len(obj))
            data = self.app.client_manager.object_store.object_create(container=parsed_args.container, object=obj, name=parsed_args.name)
            results.append(data)
        columns = ('object', 'container', 'etag')
        return (columns, (utils.get_dict_properties(s, columns, formatters={}) for s in results))