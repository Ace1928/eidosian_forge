import itertools
import json
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalDeployTemplate(command.Command):
    """Delete deploy template(s)."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalDeployTemplate')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalDeployTemplate, self).get_parser(prog_name)
        parser.add_argument('templates', metavar='<template>', nargs='+', help=_('Name(s) or UUID(s) of the deploy template(s) to delete.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for template in parsed_args.templates:
            try:
                baremetal_client.deploy_template.delete(template)
                print(_('Deleted deploy template %s') % template)
            except exc.ClientException as e:
                failures.append(_('Failed to delete deploy template %(template)s: %(error)s') % {'template': template, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))