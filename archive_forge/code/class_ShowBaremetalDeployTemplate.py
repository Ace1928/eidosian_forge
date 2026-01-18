import itertools
import json
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ShowBaremetalDeployTemplate(command.ShowOne):
    """Show baremetal deploy template details."""
    log = logging.getLogger(__name__ + '.ShowBaremetalDeployTemplate')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalDeployTemplate, self).get_parser(prog_name)
        parser.add_argument('template', metavar='<template>', help=_('Name or UUID of the deploy template.'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', choices=res_fields.DEPLOY_TEMPLATE_DETAILED_RESOURCE.fields, default=[], help=_('One or more deploy template fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        template = baremetal_client.deploy_template.get(parsed_args.template, fields=fields)._info
        template.pop('links', None)
        return zip(*sorted(template.items()))