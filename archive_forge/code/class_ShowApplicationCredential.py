import datetime
import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowApplicationCredential(command.ShowOne):
    _description = _('Display application credential details')

    def get_parser(self, prog_name):
        parser = super(ShowApplicationCredential, self).get_parser(prog_name)
        parser.add_argument('application_credential', metavar='<application-credential>', help=_('Application credential to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        app_cred = utils.find_resource(identity_client.application_credentials, parsed_args.application_credential)
        app_cred._info.pop('links', None)
        roles = app_cred._info.pop('roles')
        msg = ' '.join((r['name'] for r in roles))
        app_cred._info['roles'] = msg
        return zip(*sorted(app_cred._info.items()))