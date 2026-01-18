import logging
from osc_lib.command import command
from osc_lib import utils
from heatclient._i18n import _
from heatclient.common import format_utils
from heatclient.common import http
from heatclient.common import template_utils
from heatclient.common import utils as heat_utils
from heatclient import exc
class FunctionList(command.Lister):
    """List the available functions."""
    log = logging.getLogger(__name__ + '.FunctionList')

    def get_parser(self, prog_name):
        parser = super(FunctionList, self).get_parser(prog_name)
        parser.add_argument('template_version', metavar='<template-version>', help=_('Template version to get the functions for'))
        parser.add_argument('--with_conditions', default=False, action='store_true', help=_('Show condition functions for template.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.orchestration
        version = parsed_args.template_version
        try:
            functions = client.template_versions.get(version, with_condition_func=parsed_args.with_conditions)
        except exc.HTTPNotFound:
            msg = _('Template version not found: %s') % version
            raise exc.CommandError(msg)
        fields = ['Functions', 'Description']
        return (fields, (utils.get_item_properties(s, fields) for s in functions))