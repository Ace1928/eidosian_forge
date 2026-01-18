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