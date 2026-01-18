import itertools
import json
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class CreateBaremetalDeployTemplate(command.ShowOne):
    """Create a new deploy template"""
    log = logging.getLogger(__name__ + '.CreateBaremetalDeployTemplate')

    def get_parser(self, prog_name):
        parser = super(CreateBaremetalDeployTemplate, self).get_parser(prog_name)
        parser.add_argument('name', metavar='<name>', help=_('Unique name for this deploy template. Must be a valid trait name'))
        parser.add_argument('--uuid', dest='uuid', metavar='<uuid>', help=_('UUID of the deploy template.'))
        parser.add_argument('--extra', metavar='<key=value>', action='append', help=_('Record arbitrary key/value metadata. Can be specified multiple times.'))
        parser.add_argument('--steps', metavar='<steps>', required=True, help=_DEPLOY_STEPS_HELP)
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        steps = utils.handle_json_arg(parsed_args.steps, 'deploy steps')
        field_list = ['name', 'uuid', 'extra']
        fields = dict(((k, v) for k, v in vars(parsed_args).items() if k in field_list and v is not None))
        fields = utils.args_array_to_dict(fields, 'extra')
        template = baremetal_client.deploy_template.create(steps=steps, **fields)
        data = dict([(f, getattr(template, f, '')) for f in res_fields.DEPLOY_TEMPLATE_DETAILED_RESOURCE.fields])
        return self.dict2columns(data)