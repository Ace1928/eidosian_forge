from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
class ParametersBase(common.HeatTestCase):

    def new_parameters(self, stack_name, tmpl, user_params=None, stack_id=None, validate_value=True, param_defaults=None):
        user_params = user_params or {}
        tmpl.update({'HeatTemplateFormatVersion': '2012-12-12'})
        tmpl = template.Template(tmpl)
        params = tmpl.parameters(identifier.HeatIdentifier('', stack_name, stack_id), user_params, param_defaults=param_defaults)
        params.validate(validate_value)
        return params