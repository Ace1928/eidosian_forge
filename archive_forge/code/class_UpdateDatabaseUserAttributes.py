from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class UpdateDatabaseUserAttributes(command.Command):
    _description = _("Updates a user's attributes on an instance.At least one optional argument must be provided.")

    def get_parser(self, prog_name):
        parser = super(UpdateDatabaseUserAttributes, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of user.'))
        parser.add_argument('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
        parser.add_argument('--new_name', metavar='<new_name>', default=None, help=_('Optional new name of user.'))
        parser.add_argument('--new_password', metavar='<new_password>', default=None, help=_('Optional new password of user.'))
        parser.add_argument('--new_host', metavar='<new_host>', default=None, help=_('Optional new host of user.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        users = manager.users
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        new_attrs = {}
        if parsed_args.new_name:
            new_attrs['name'] = parsed_args.new_name
        if parsed_args.new_password:
            new_attrs['password'] = parsed_args.new_password
        if parsed_args.new_host:
            new_attrs['host'] = parsed_args.new_host
        users.update_attributes(instance, parsed_args.name, newuserattr=new_attrs, hostname=parsed_args.host)