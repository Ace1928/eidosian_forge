import argparse
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.common import exceptions as nc_exc
class SetSfcFlowClassifier(command.Command):
    _description = _('Set flow classifier properties')

    def get_parser(self, prog_name):
        parser = super(SetSfcFlowClassifier, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('Name of the flow classifier'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the flow classifier'))
        parser.add_argument('flow_classifier', metavar='<flow-classifier>', help=_('Flow classifier to modify (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fc_id = client.find_sfc_flow_classifier(parsed_args.flow_classifier, ignore_missing=False)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        try:
            client.update_sfc_flow_classifier(fc_id, **attrs)
        except Exception as e:
            msg = _("Failed to update flow classifier '%(fc)s': %(e)s") % {'fc': parsed_args.flow_classifier, 'e': e}
            raise exceptions.CommandError(msg)