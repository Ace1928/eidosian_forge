import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
class SetSfcPortChain(command.Command):
    _description = _('Set port chain properties')

    def get_parser(self, prog_name):
        parser = super(SetSfcPortChain, self).get_parser(prog_name)
        parser.add_argument('--name', metavar='<name>', help=_('Name of the port chain'))
        parser.add_argument('--description', metavar='<description>', help=_('Description for the port chain'))
        parser.add_argument('--flow-classifier', metavar='<flow-classifier>', dest='flow_classifiers', action='append', help=_('Add flow classifier (name or ID). This option can be repeated.'))
        parser.add_argument('--no-flow-classifier', action='store_true', help=_('Remove associated flow classifiers from the port chain'))
        parser.add_argument('--port-pair-group', metavar='<port-pair-group>', dest='port_pair_groups', action='append', help=_('Add port pair group (name or ID). Current port pair groups order is kept, the added port pair group will be placed at the end of the port chain. This option can be repeated.'))
        parser.add_argument('--no-port-pair-group', action='store_true', help=_('Remove associated port pair groups from the port chain. At least one --port-pair-group must be specified together.'))
        parser.add_argument('port_chain', metavar='<port-chain>', help=_('Port chain to modify (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        pc_id = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['id']
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        if parsed_args.no_flow_classifier:
            attrs['flow_classifiers'] = []
        if parsed_args.flow_classifiers:
            if parsed_args.no_flow_classifier:
                fc_list = []
            else:
                fc_list = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['flow_classifiers']
            for fc in parsed_args.flow_classifiers:
                fc_id = client.find_sfc_flow_classifier(fc, ignore_missing=False)['id']
                if fc_id not in fc_list:
                    fc_list.append(fc_id)
            attrs['flow_classifiers'] = fc_list
        if parsed_args.no_port_pair_group and (not parsed_args.port_pair_groups):
            message = _('At least one --port-pair-group must be specified.')
            raise exceptions.CommandError(message)
        if parsed_args.no_port_pair_group and parsed_args.port_pair_groups:
            ppg_list = []
            for ppg in parsed_args.port_pair_groups:
                ppg_id = client.find_sfc_port_pair_group(ppg, ignore_missing=False)['id']
                if ppg_id not in ppg_list:
                    ppg_list.append(ppg_id)
            attrs['port_pair_groups'] = ppg_list
        if parsed_args.port_pair_groups and (not parsed_args.no_port_pair_group):
            ppg_list = client.find_sfc_port_chain(parsed_args.port_chain, ignore_missing=False)['port_pair_groups']
            for ppg in parsed_args.port_pair_groups:
                ppg_id = client.find_sfc_port_pair_group(ppg, ignore_missing=False)['id']
                if ppg_id not in ppg_list:
                    ppg_list.append(ppg_id)
            attrs['port_pair_groups'] = ppg_list
        try:
            client.update_sfc_port_chain(pc_id, **attrs)
        except Exception as e:
            msg = _("Failed to update port chain '%(pc)s': %(e)s") % {'pc': parsed_args.port_chain, 'e': e}
            raise exceptions.CommandError(msg)