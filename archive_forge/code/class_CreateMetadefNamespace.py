import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateMetadefNamespace(command.ShowOne):
    _description = _('Create a metadef namespace')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('New metadef namespace name'))
        parser.add_argument('--display-name', metavar='<display_name>', help=_('A user-friendly name for the namespace.'))
        parser.add_argument('--description', metavar='<description>', help=_('A description of the namespace'))
        visibility_group = parser.add_mutually_exclusive_group()
        visibility_group.add_argument('--public', action='store_const', const='public', dest='visibility', help=_("Set namespace visibility 'public'"))
        visibility_group.add_argument('--private', action='store_const', const='private', dest='visibility', help=_("Set namespace visibility 'private'"))
        protected_group = parser.add_mutually_exclusive_group()
        protected_group.add_argument('--protected', action='store_const', const=True, dest='is_protected', help=_('Prevent metadef namespace from being deleted'))
        protected_group.add_argument('--unprotected', action='store_const', const=False, dest='is_protected', help=_('Allow metadef namespace to be deleted (default)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        filter_keys = ['namespace', 'display_name', 'description']
        kwargs = {}
        for key in filter_keys:
            argument = getattr(parsed_args, key, None)
            if argument is not None:
                kwargs[key] = argument
        if parsed_args.is_protected is not None:
            kwargs['protected'] = parsed_args.is_protected
        if parsed_args.visibility is not None:
            kwargs['visibility'] = parsed_args.visibility
        data = image_client.create_metadef_namespace(**kwargs)
        info = _format_namespace(data)
        return zip(*sorted(info.items()))