import configparser
import re
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import policy
import glance.api.policy
from glance.common import exception
from glance.i18n import _, _LE, _LW
def _load_rules(self):
    try:
        conf_file = CONF.find_file(CONF.property_protection_file)
        CONFIG.read(conf_file)
    except Exception as e:
        msg = _LE("Couldn't find property protection file %(file)s: %(error)s.") % {'file': CONF.property_protection_file, 'error': e}
        LOG.error(msg)
        raise InvalidPropProtectConf()
    if self.prop_prot_rule_format not in ['policies', 'roles']:
        msg = _LE("Invalid value '%s' for 'property_protection_rule_format'. The permitted values are 'roles' and 'policies'") % self.prop_prot_rule_format
        LOG.error(msg)
        raise InvalidPropProtectConf()
    operations = ['create', 'read', 'update', 'delete']
    properties = CONFIG.sections()
    for property_exp in properties:
        property_dict = {}
        compiled_rule = self._compile_rule(property_exp)
        for operation in operations:
            try:
                permissions = CONFIG.get(property_exp, operation)
            except configparser.NoOptionError:
                raise InvalidPropProtectConf()
            if permissions:
                if self.prop_prot_rule_format == 'policies':
                    if ',' in permissions:
                        LOG.error(_LE("Multiple policies '%s' not allowed for a given operation. Policies can be combined in the policy file"), permissions)
                        raise InvalidPropProtectConf()
                    self.prop_exp_mapping[compiled_rule] = property_exp
                    self._add_policy_rules(property_exp, operation, permissions)
                    permissions = [permissions]
                else:
                    permissions = [permission.strip() for permission in permissions.split(',')]
                    if '@' in permissions and '!' in permissions:
                        msg = _LE("Malformed property protection rule in [%(prop)s] %(op)s=%(perm)s: '@' and '!' are mutually exclusive") % dict(prop=property_exp, op=operation, perm=permissions)
                        LOG.error(msg)
                        raise InvalidPropProtectConf()
                property_dict[operation] = permissions
            else:
                property_dict[operation] = []
                LOG.warning(_LW('Property protection on operation %(operation)s for rule %(rule)s is not found. No role will be allowed to perform this operation.'), {'operation': operation, 'rule': property_exp})
        self.rules.append((compiled_rule, property_dict))