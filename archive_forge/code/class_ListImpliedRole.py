import logging
from osc_lib.command import command
from openstackclient.i18n import _
class ListImpliedRole(command.Lister):
    _description = _('List implied roles')
    _COLUMNS = ['Prior Role ID', 'Prior Role Name', 'Implied Role ID', 'Implied Role Name']

    def get_parser(self, prog_name):
        parser = super(ListImpliedRole, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):

        def _list_implied(response):
            for rule in response:
                for implies in rule.implies:
                    yield (rule.prior_role['id'], rule.prior_role['name'], implies['id'], implies['name'])
        identity_client = self.app.client_manager.identity
        response = identity_client.inference_rules.list_inference_roles()
        return (self._COLUMNS, _list_implied(response))