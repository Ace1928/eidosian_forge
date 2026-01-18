import sys
from cliff import command
from cliff import lister
from cliff import show
from oslo_log import log
from vitrageclient.common import utils
from vitrageclient.common.utils import find_template_with_uuid
class TemplateValidate(show.ShowOne):
    """Validate a template file"""

    def get_parser(self, prog_name):
        parser = super(TemplateValidate, self).get_parser(prog_name)
        parser.add_argument('--path', required=True, help='full path for template file or templates dir')
        parser.add_argument('--type', choices=['standard', 'definition', 'equivalence'], help='Template type. Valid types:[standard, definition, equivalence]')
        parser.add_argument('--params', nargs='+', help="Actual values for parameters of the template. Several key=value pairs may be used, for example: --params template_name=cpu_problem alarm_name='High CPU Load'")
        return parser

    @property
    def formatter_default(self):
        return 'json'

    def take_action(self, parsed_args):
        cli_param_list = parsed_args.params
        params = _parse_template_params(cli_param_list)
        result = utils.get_client(self).template.validate(path=parsed_args.path, template_type=parsed_args.type, params=params)
        return self.dict2columns(result)