import sys
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListModule(command.ShowOne):
    _description = _('List module versions')
    auth_required = False

    def get_parser(self, prog_name):
        parser = super(ListModule, self).get_parser(prog_name)
        parser.add_argument('--all', action='store_true', default=False, help=_('Show all modules that have version information'))
        return parser

    def take_action(self, parsed_args):
        data = {}
        mods = sys.modules
        for k in mods.keys():
            k = k.split('.')[0]
            if not k.startswith('_') and k not in data:
                if parsed_args.all or (k.endswith('client') or k == 'openstack'):
                    try:
                        if k == 'openstack':
                            data[k] = mods[k].version.__version__
                        else:
                            data[k] = mods[k].__version__
                    except Exception:
                        pass
        return zip(*sorted(data.items()))