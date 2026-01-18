import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
def _find_group():
    try:
        return common.find_group(identity_client_manager, parsed_args.group, parsed_args.group_domain).id
    except exceptions.CommandError:
        if not validate_actor_existence:
            return parsed_args.group
        raise