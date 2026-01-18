import collections
import copy
import functools
from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.heat import software_config as sc
from heat.engine.resources.openstack.heat import software_deployment as sd
from heat.engine import rsrc_defn
from heat.engine import support
class StructuredDeployment(sd.SoftwareDeployment):
    """A resource which has same logic with OS::Heat::SoftwareDeployment.

    A deployment resource like OS::Heat::SoftwareDeployment, but which
    performs input value substitution on the config defined by a
    OS::Heat::StructuredConfig resource.

    Some configuration tools have no concept of inputs, so the input value
    substitution needs to occur in the deployment resource. An example of this
    is the JSON metadata consumed by the cfn-init tool.

    Where the config contains {get_input: input_name} this will be substituted
    with the value of input_name in this resource's input_values. If get_input
    needs to be passed through to the substituted configuration then a
    different input_key property value can be specified.
    """
    support_status = support.SupportStatus(version='2014.1')
    PROPERTIES = CONFIG, SERVER, INPUT_VALUES, DEPLOY_ACTIONS, NAME, SIGNAL_TRANSPORT, INPUT_KEY, INPUT_VALUES_VALIDATE = (sd.SoftwareDeployment.CONFIG, sd.SoftwareDeployment.SERVER, sd.SoftwareDeployment.INPUT_VALUES, sd.SoftwareDeployment.DEPLOY_ACTIONS, sd.SoftwareDeployment.NAME, sd.SoftwareDeployment.SIGNAL_TRANSPORT, 'input_key', 'input_values_validate')
    _sd_ps = sd.SoftwareDeployment.properties_schema
    properties_schema = {CONFIG: _sd_ps[CONFIG], SERVER: _sd_ps[SERVER], INPUT_VALUES: _sd_ps[INPUT_VALUES], DEPLOY_ACTIONS: _sd_ps[DEPLOY_ACTIONS], SIGNAL_TRANSPORT: _sd_ps[SIGNAL_TRANSPORT], NAME: _sd_ps[NAME], INPUT_KEY: properties.Schema(properties.Schema.STRING, _('Name of key to use for substituting inputs during deployment.'), default='get_input'), INPUT_VALUES_VALIDATE: properties.Schema(properties.Schema.STRING, _('Perform a check on the input values passed to verify that each required input has a corresponding value. When the property is set to STRICT and no value is passed, an exception is raised.'), default='LAX', constraints=[constraints.AllowedValues(['LAX', 'STRICT'])])}

    def empty_config(self):
        return {}

    def _build_derived_config(self, action, source, derived_inputs, derived_options):
        cfg = source.get(sc.SoftwareConfig.CONFIG)
        input_key = self.properties[self.INPUT_KEY]
        check_input_val = self.properties[self.INPUT_VALUES_VALIDATE]
        inputs = dict((i.input_data() for i in derived_inputs))
        return self.parse(inputs, input_key, cfg, check_input_val)

    @staticmethod
    def get_input_key_arg(snippet, input_key):
        if len(snippet) != 1:
            return None
        fn_name, fn_arg = next(iter(snippet.items()))
        if fn_name == input_key and isinstance(fn_arg, str):
            return fn_arg

    @staticmethod
    def get_input_key_value(fn_arg, inputs, check_input_val='LAX'):
        if check_input_val == 'STRICT' and fn_arg not in inputs:
            raise exception.UserParameterMissing(key=fn_arg)
        return inputs.get(fn_arg)

    @staticmethod
    def parse(inputs, input_key, snippet, check_input_val='LAX'):
        parse = functools.partial(StructuredDeployment.parse, inputs, input_key, check_input_val=check_input_val)
        if isinstance(snippet, collections.abc.Mapping):
            fn_arg = StructuredDeployment.get_input_key_arg(snippet, input_key)
            if fn_arg is not None:
                return StructuredDeployment.get_input_key_value(fn_arg, inputs, check_input_val)
            return dict(((k, parse(v)) for k, v in snippet.items()))
        elif not isinstance(snippet, str) and isinstance(snippet, collections.abc.Iterable):
            return [parse(v) for v in snippet]
        else:
            return snippet