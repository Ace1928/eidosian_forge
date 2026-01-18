import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class DefaultDatabaseConfiguration(command.ShowOne):
    _description = _('Shows the default configuration of an instance.')

    def get_parser(self, prog_name):
        parser = super(DefaultDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        configs = db_instances.configuration(instance)
        return zip(*sorted(configs._info['configuration'].items()))