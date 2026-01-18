from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
class DomainConfigManager(manager.Manager):
    """Default pivot point for the Domain Config backend."""
    driver_namespace = 'keystone.resource.domain_config'
    _provides_api = 'domain_config_api'
    whitelisted_options = {'identity': ['driver', 'list_limit'], 'ldap': ['url', 'user', 'suffix', 'query_scope', 'page_size', 'alias_dereferencing', 'debug_level', 'chase_referrals', 'user_tree_dn', 'user_filter', 'user_objectclass', 'user_id_attribute', 'user_name_attribute', 'user_mail_attribute', 'user_description_attribute', 'user_pass_attribute', 'user_enabled_attribute', 'user_enabled_invert', 'user_enabled_mask', 'user_enabled_default', 'user_attribute_ignore', 'user_default_project_id_attribute', 'user_enabled_emulation', 'user_enabled_emulation_dn', 'user_enabled_emulation_use_group_config', 'user_additional_attribute_mapping', 'group_tree_dn', 'group_filter', 'group_objectclass', 'group_id_attribute', 'group_name_attribute', 'group_members_are_ids', 'group_member_attribute', 'group_desc_attribute', 'group_attribute_ignore', 'group_additional_attribute_mapping', 'tls_cacertfile', 'tls_cacertdir', 'use_tls', 'tls_req_cert', 'use_pool', 'pool_size', 'pool_retry_max', 'pool_retry_delay', 'pool_connection_timeout', 'pool_connection_lifetime', 'use_auth_pool', 'auth_pool_size', 'auth_pool_connection_lifetime']}
    sensitive_options = {'identity': [], 'ldap': ['password']}

    def __init__(self):
        super(DomainConfigManager, self).__init__(CONF.domain_config.driver)

    def _assert_valid_config(self, config):
        """Ensure the options in the config are valid.

        This method is called to validate the request config in create and
        update manager calls.

        :param config: config structure being created or updated

        """
        if not config:
            raise exception.InvalidDomainConfig(reason=_('No options specified'))
        for group in config:
            if not config[group] or not isinstance(config[group], dict):
                msg = _('The value of group %(group)s specified in the config should be a dictionary of options') % {'group': group}
                raise exception.InvalidDomainConfig(reason=msg)
            for option in config[group]:
                self._assert_valid_group_and_option(group, option)

    def _assert_valid_group_and_option(self, group, option):
        """Ensure the combination of group and option is valid.

        :param group: optional group name, if specified it must be one
                      we support
        :param option: optional option name, if specified it must be one
                       we support and a group must also be specified

        """
        if not group and (not option):
            return
        if not group and option:
            msg = _('Option %(option)s found with no group specified while checking domain configuration request') % {'option': option}
            raise exception.UnexpectedError(exception=msg)
        if group and group not in self.whitelisted_options and (group not in self.sensitive_options):
            msg = _('Group %(group)s is not supported for domain specific configurations') % {'group': group}
            raise exception.InvalidDomainConfig(reason=msg)
        if option:
            if option not in self.whitelisted_options[group] and option not in self.sensitive_options[group]:
                msg = _('Option %(option)s in group %(group)s is not supported for domain specific configurations') % {'group': group, 'option': option}
                raise exception.InvalidDomainConfig(reason=msg)

    def _is_sensitive(self, group, option):
        return option in self.sensitive_options[group]

    def _config_to_list(self, config):
        """Build list of options for use by backend drivers."""
        option_list = []
        for group in config:
            for option in config[group]:
                option_list.append({'group': group, 'option': option, 'value': config[group][option], 'sensitive': self._is_sensitive(group, option)})
        return option_list

    def _option_dict(self, group, option):
        group_attr = getattr(CONF, group)
        return {'group': group, 'option': option, 'value': getattr(group_attr, option)}

    def _list_to_config(self, whitelisted, sensitive=None, req_option=None):
        """Build config dict from a list of option dicts.

        :param whitelisted: list of dicts containing options and their groups,
                            this has already been filtered to only contain
                            those options to include in the output.
        :param sensitive: list of dicts containing sensitive options and their
                          groups, this has already been filtered to only
                          contain those options to include in the output.
        :param req_option: the individual option requested

        :returns: a config dict, including sensitive if specified

        """
        the_list = whitelisted + (sensitive or [])
        if not the_list:
            return {}
        if req_option:
            if len(the_list) > 1 or the_list[0]['option'] != req_option:
                LOG.error('Unexpected results in response for domain config - %(count)s responses, first option is %(option)s, expected option %(expected)s', {'count': len(the_list), 'option': list[0]['option'], 'expected': req_option})
                raise exception.UnexpectedError(_('An unexpected error occurred when retrieving domain configs'))
            return {the_list[0]['option']: the_list[0]['value']}
        config = {}
        for option in the_list:
            config.setdefault(option['group'], {})
            config[option['group']][option['option']] = option['value']
        return config

    def create_config(self, domain_id, config):
        """Create config for a domain.

        :param domain_id: the domain in question
        :param config: the dict of config groups/options to assign to the
                       domain

        Creates a new config, overwriting any previous config (no Conflict
        error will be generated).

        :returns: a dict of group dicts containing the options, with any that
                  are sensitive removed
        :raises keystone.exception.InvalidDomainConfig: when the config
                contains options we do not support

        """
        self._assert_valid_config(config)
        option_list = self._config_to_list(config)
        self.create_config_options(domain_id, option_list)
        self.get_config_with_sensitive_info.invalidate(self, domain_id)
        return self._list_to_config(self.list_config_options(domain_id))

    def get_config(self, domain_id, group=None, option=None):
        """Get config, or partial config, for a domain.

        :param domain_id: the domain in question
        :param group: an optional specific group of options
        :param option: an optional specific option within the group

        :returns: a dict of group dicts containing the whitelisted options,
                  filtered by group and option specified
        :raises keystone.exception.DomainConfigNotFound: when no config found
                that matches domain_id, group and option specified
        :raises keystone.exception.InvalidDomainConfig: when the config
                and group/option parameters specify an option we do not
                support

        An example response::

            {
                'ldap': {
                    'url': 'myurl'
                    'user_tree_dn': 'OU=myou'},
                'identity': {
                    'driver': 'ldap'}

            }

        """
        self._assert_valid_group_and_option(group, option)
        whitelisted = self.list_config_options(domain_id, group, option)
        if whitelisted:
            return self._list_to_config(whitelisted, req_option=option)
        if option:
            msg = _('option %(option)s in group %(group)s') % {'group': group, 'option': option}
        elif group:
            msg = _('group %(group)s') % {'group': group}
        else:
            msg = _('any options')
        raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)

    def get_security_compliance_config(self, domain_id, group, option=None):
        """Get full or partial security compliance config from configuration.

        :param domain_id: the domain in question
        :param group: a specific group of options
        :param option: an optional specific option within the group

        :returns: a dict of group dicts containing the whitelisted options,
                  filtered by group and option specified
        :raises keystone.exception.InvalidDomainConfig: when the config
                and group/option parameters specify an option we do not
                support

        An example response::

            {
                'security_compliance': {
                    'password_regex': '^(?=.*\\d)(?=.*[a-zA-Z]).{7,}$'
                    'password_regex_description':
                        'A password must consist of at least 1 letter, '
                        '1 digit, and have a minimum length of 7 characters'
                    }
            }

        """
        if domain_id != CONF.identity.default_domain_id:
            msg = _('Reading security compliance information for any domain other than the default domain is not allowed or supported.')
            raise exception.InvalidDomainConfig(reason=msg)
        config_list = []
        readable_options = ['password_regex', 'password_regex_description']
        if option and option not in readable_options:
            msg = _('Reading security compliance values other than password_regex and password_regex_description is not allowed.')
            raise exception.InvalidDomainConfig(reason=msg)
        elif option and option in readable_options:
            config_list.append(self._option_dict(group, option))
        elif not option:
            for op in readable_options:
                config_list.append(self._option_dict(group, op))
        return self._list_to_config(config_list, req_option=option)

    def update_config(self, domain_id, config, group=None, option=None):
        """Update config, or partial config, for a domain.

        :param domain_id: the domain in question
        :param config: the config dict containing and groups/options being
                       updated
        :param group: an optional specific group of options, which if specified
                      must appear in config, with no other groups
        :param option: an optional specific option within the group, which if
                       specified must appear in config, with no other options

        The contents of the supplied config will be merged with the existing
        config for this domain, updating or creating new options if these did
        not previously exist. If group or option is specified, then the update
        will be limited to those specified items and the inclusion of other
        options in the supplied config will raise an exception, as will the
        situation when those options do not already exist in the current
        config.

        :returns: a dict of groups containing all whitelisted options
        :raises keystone.exception.InvalidDomainConfig: when the config
                and group/option parameters specify an option we do not
                support or one that does not exist in the original config

        """

        def _assert_valid_update(domain_id, config, group=None, option=None):
            """Ensure the combination of config, group and option is valid."""
            self._assert_valid_config(config)
            self._assert_valid_group_and_option(group, option)
            if group:
                if len(config) != 1 or (option and len(config[group]) != 1):
                    if option:
                        msg = _('Trying to update option %(option)s in group %(group)s, so that, and only that, option must be specified  in the config') % {'group': group, 'option': option}
                    else:
                        msg = _('Trying to update group %(group)s, so that, and only that, group must be specified in the config') % {'group': group}
                    raise exception.InvalidDomainConfig(reason=msg)
                if group not in config:
                    msg = _('request to update group %(group)s, but config provided contains group %(group_other)s instead') % {'group': group, 'group_other': list(config.keys())[0]}
                    raise exception.InvalidDomainConfig(reason=msg)
                if option and option not in config[group]:
                    msg = _('Trying to update option %(option)s in group %(group)s, but config provided contains option %(option_other)s instead') % {'group': group, 'option': option, 'option_other': list(config[group].keys())[0]}
                    raise exception.InvalidDomainConfig(reason=msg)
                if not self._get_config_with_sensitive_info(domain_id, group, option):
                    if option:
                        msg = _('option %(option)s in group %(group)s') % {'group': group, 'option': option}
                        raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
                    else:
                        msg = _('group %(group)s') % {'group': group}
                        raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
        update_config = config
        if group and option:
            update_config = {group: config}
        _assert_valid_update(domain_id, update_config, group, option)
        option_list = self._config_to_list(update_config)
        self.update_config_options(domain_id, option_list)
        self.get_config_with_sensitive_info.invalidate(self, domain_id)
        return self.get_config(domain_id)

    def delete_config(self, domain_id, group=None, option=None):
        """Delete config, or partial config, for the domain.

        :param domain_id: the domain in question
        :param group: an optional specific group of options
        :param option: an optional specific option within the group

        If group and option are None, then the entire config for the domain
        is deleted. If group is not None, then just that group of options will
        be deleted. If group and option are both specified, then just that
        option is deleted.

        :raises keystone.exception.InvalidDomainConfig: when group/option
                parameters specify an option we do not support or one that
                does not exist in the original config.

        """
        self._assert_valid_group_and_option(group, option)
        if group:
            current_config = self._get_config_with_sensitive_info(domain_id)
            current_group = current_config.get(group)
            if not current_group:
                msg = _('group %(group)s') % {'group': group}
                raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
            if option and (not current_group.get(option)):
                msg = _('option %(option)s in group %(group)s') % {'group': group, 'option': option}
                raise exception.DomainConfigNotFound(domain_id=domain_id, group_or_option=msg)
        self.delete_config_options(domain_id, group, option)
        self.get_config_with_sensitive_info.invalidate(self, domain_id)

    def _get_config_with_sensitive_info(self, domain_id, group=None, option=None):
        """Get config for a domain/group/option with sensitive info included.

        This is only used by the methods within this class, which may need to
        check individual groups or options.

        """
        whitelisted = self.list_config_options(domain_id, group, option)
        sensitive = self.list_config_options(domain_id, group, option, sensitive=True)
        sensitive_dict = {s['option']: s['value'] for s in sensitive}
        for each_whitelisted in whitelisted:
            if not isinstance(each_whitelisted['value'], str):
                continue
            original_value = each_whitelisted['value']
            warning_msg = ''
            try:
                each_whitelisted['value'] = each_whitelisted['value'] % sensitive_dict
            except KeyError:
                warning_msg = 'Found what looks like an unmatched config option substitution reference - domain: %(domain)s, group: %(group)s, option: %(option)s, value: %(value)s. Perhaps the config option to which it refers has yet to be added?'
            except (ValueError, TypeError):
                warning_msg = 'Found what looks like an incorrectly constructed config option substitution reference - domain: %(domain)s, group: %(group)s, option: %(option)s, value: %(value)s.'
            if warning_msg:
                LOG.warning(warning_msg, {'domain': domain_id, 'group': each_whitelisted['group'], 'option': each_whitelisted['option'], 'value': original_value})
        return self._list_to_config(whitelisted, sensitive)

    @MEMOIZE_CONFIG
    def get_config_with_sensitive_info(self, domain_id):
        """Get config for a domain with sensitive info included.

        This method is not exposed via the public API, but is used by the
        identity manager to initialize a domain with the fully formed config
        options.

        """
        return self._get_config_with_sensitive_info(domain_id)

    def get_config_default(self, group=None, option=None):
        """Get default config, or partial default config.

        :param group: an optional specific group of options
        :param option: an optional specific option within the group

        :returns: a dict of group dicts containing the default options,
                  filtered by group and option if specified
        :raises keystone.exception.InvalidDomainConfig: when the config
                and group/option parameters specify an option we do not
                support (or one that is not whitelisted).

        An example response::

            {
                'ldap': {
                    'url': 'myurl',
                    'user_tree_dn': 'OU=myou',
                    ....},
                'identity': {
                    'driver': 'ldap'}

            }

        """
        self._assert_valid_group_and_option(group, option)
        config_list = []
        if group:
            if option:
                if option not in self.whitelisted_options[group]:
                    msg = _('Reading the default for option %(option)s in group %(group)s is not supported') % {'option': option, 'group': group}
                    raise exception.InvalidDomainConfig(reason=msg)
                config_list.append(self._option_dict(group, option))
            else:
                for each_option in self.whitelisted_options[group]:
                    config_list.append(self._option_dict(group, each_option))
        else:
            for each_group in self.whitelisted_options:
                for each_option in self.whitelisted_options[each_group]:
                    config_list.append(self._option_dict(each_group, each_option))
        return self._list_to_config(config_list, req_option=option)