from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient.i18n import _
from troveclient import utils
class ListDatabaseLimits(command.Lister):
    _description = _('List database limits')
    columns = ['Value', 'Verb', 'Remaining', 'Unit']

    def take_action(self, parsed_args):
        database_limits = self.app.client_manager.database.limits
        limits = database_limits.list()
        utils.print_dict(limits.pop(0)._info)
        limits = [osc_utils.get_item_properties(i, self.columns) for i in limits]
        return (self.columns, limits)