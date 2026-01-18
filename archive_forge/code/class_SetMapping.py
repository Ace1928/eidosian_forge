import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetMapping(command.Command, _RulesReader):
    _description = _('Set mapping properties')

    def get_parser(self, prog_name):
        parser = super(SetMapping, self).get_parser(prog_name)
        parser.add_argument('mapping', metavar='<name>', help=_('Mapping to modify'))
        parser.add_argument('--rules', metavar='<filename>', help=_('Filename that contains a new set of mapping rules'))
        _RulesReader.add_federated_schema_version_option(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        rules = self._read_rules(parsed_args.rules)
        mapping = identity_client.federation.mappings.update(mapping=parsed_args.mapping, rules=rules, schema_version=parsed_args.schema_version)
        mapping._info.pop('links', None)