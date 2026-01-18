from webob import exc
from neutron_lib._i18n import _
from neutron_lib.api.definitions import network
from neutron_lib.api.definitions import port
from neutron_lib.api.definitions import subnet
from neutron_lib.api.definitions import subnetpool
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions
def _dict_populate_defaults(attr_value, attr_spec):
    if not attr_spec.get(constants.DICT_POPULATE_DEFAULTS):
        return attr_value
    if attr_value is None or attr_value is constants.ATTR_NOT_SPECIFIED:
        attr_value = {}
    for rule_type, rule_content in attr_spec['validate'].items():
        if 'dict' not in rule_type:
            continue
        for key, key_validator in rule_content.items():
            validator_name, _dummy, validator_params = validators._extract_validator(key_validator)
            if 'dict' in validator_name:
                value = _dict_populate_defaults(attr_value.get(key), {constants.DICT_POPULATE_DEFAULTS: key_validator.get(constants.DICT_POPULATE_DEFAULTS), 'validate': {validator_name: validator_params}})
                if value is not None:
                    attr_value[key] = value
            _fill_default(attr_value, key, key_validator)
    return attr_value