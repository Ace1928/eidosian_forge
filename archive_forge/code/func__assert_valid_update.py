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