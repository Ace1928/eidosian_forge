import argparse
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import exceptions as nc_exc
class DeleteSfcFlowClassifier(command.Command):
    _description = _('Delete a given flow classifier')

    def get_parser(self, prog_name):
        parser = super(DeleteSfcFlowClassifier, self).get_parser(prog_name)
        parser.add_argument('flow_classifier', metavar='<flow-classifier>', nargs='+', help=_('Flow classifier(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for fcl in parsed_args.flow_classifier:
            try:
                fc_id = client.find_sfc_flow_classifier(fcl, ignore_missing=False)['id']
                client.delete_sfc_flow_classifier(fc_id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete flow classifier with name or ID '%(fc)s': %(e)s"), {'fc': fcl, 'e': e})
        if result > 0:
            total = len(parsed_args.flow_classifier)
            msg = _('%(result)s of %(total)s flow classifier(s) failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)