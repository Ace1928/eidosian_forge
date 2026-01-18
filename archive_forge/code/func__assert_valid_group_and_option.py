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