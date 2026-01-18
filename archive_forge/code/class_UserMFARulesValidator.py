from functools import partial
from oslo_log import log
import stevedore
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import resource_options as ro
class UserMFARulesValidator(provider_api.ProviderAPIMixin, object):
    """Helper object that can validate the MFA Rules."""

    @classmethod
    def _auth_methods(cls):
        if AUTH_PLUGINS_LOADED:
            return set(AUTH_METHODS.keys())
        raise RuntimeError(_('Auth Method Plugins are not loaded.'))

    @classmethod
    def check_auth_methods_against_rules(cls, user_id, auth_methods):
        """Validate the MFA rules against the successful auth methods.

        :param user_id: The user's ID (uuid).
        :type user_id: str
        :param auth_methods: List of methods that were used for auth
        :type auth_methods: set
        :returns: Boolean, ``True`` means rules match and auth may proceed,
                  ``False`` means rules do not match.
        """
        user_ref = PROVIDERS.identity_api.get_user(user_id)
        mfa_rules = user_ref['options'].get(ro.MFA_RULES_OPT.option_name, [])
        mfa_rules_enabled = user_ref['options'].get(ro.MFA_ENABLED_OPT.option_name, True)
        rules = cls._parse_rule_structure(mfa_rules, user_ref['id'])
        if not rules or not mfa_rules_enabled:
            LOG.debug('MFA Rules not processed for user `%(user_id)s`. Rule list: `%(rules)s` (Enabled: `%(enabled)s`).', {'user_id': user_id, 'rules': mfa_rules, 'enabled': mfa_rules_enabled})
            return True
        for r in rules:
            r_set = set(r).intersection(cls._auth_methods())
            if set(auth_methods).issuperset(r_set):
                LOG.debug('Auth methods for user `%(user_id)s`, `%(methods)s` matched MFA rule `%(rule)s`. Loaded auth_methods: `%(loaded)s`', {'user_id': user_id, 'rule': list(r_set), 'methods': auth_methods, 'loaded': cls._auth_methods()})
                return True
        LOG.debug('Auth methods for user `%(user_id)s`, `%(methods)s` did not match a MFA rule in `%(rules)s`.', {'user_id': user_id, 'methods': auth_methods, 'rules': rules})
        return False

    @staticmethod
    def _parse_rule_structure(rules, user_id):
        """Validate and parse the rule data structure.

        Rule sets must be in the form of list of lists. The lists may not
        have duplicates and must not be empty. The top-level list may be empty
        indicating that no rules exist.

        :param rules: The list of rules from the user_ref
        :type rules: list
        :param user_id: the user_id, used for logging purposes
        :type user_id: str
        :returns: list of list, duplicates are stripped
        """
        rule_set = []
        if not isinstance(rules, list):
            LOG.error('Corrupt rule data structure for user %(user_id)s, no rules loaded.', {'user_id': user_id})
            return rule_set
        elif not rules:
            return rule_set
        for r_list in rules:
            if not isinstance(r_list, list):
                LOG.info('Ignoring Rule %(type)r; rule must be a list of strings.', {'type': type(r_list)})
                continue
            if r_list:
                _ok_rule = True
                for item in r_list:
                    if not isinstance(item, str):
                        LOG.info('Ignoring Rule %(rule)r; rule contains non-string values.', {'rule': r_list})
                        _ok_rule = False
                        break
                if _ok_rule:
                    r_list = list(set(r_list))
                    rule_set.append(r_list)
        return rule_set