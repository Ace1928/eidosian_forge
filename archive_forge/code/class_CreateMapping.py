import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateMapping(command.ShowOne, _RulesReader):
    _description = _('Create new mapping')

    def get_parser(self, prog_name):
        parser = super(CreateMapping, self).get_parser(prog_name)
        parser.add_argument('mapping', metavar='<name>', help=_('New mapping name (must be unique)'))
        parser.add_argument('--rules', metavar='<filename>', required=True, help=_('Filename that contains a set of mapping rules (required)'))
        _RulesReader.add_federated_schema_version_option(parser)
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        rules = self._read_rules(parsed_args.rules)
        mapping = identity_client.federation.mappings.create(mapping_id=parsed_args.mapping, rules=rules, schema_version=parsed_args.schema_version)
        mapping._info.pop('links', None)
        return zip(*sorted(mapping._info.items()))