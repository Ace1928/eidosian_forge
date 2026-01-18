import logging
from keystoneauth1 import exceptions as ks_exc
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
def _process_identity_and_resource_options(parsed_args, identity_client_manager, validate_actor_existence=True):

    def _find_user():
        try:
            return common.find_user(identity_client_manager, parsed_args.user, parsed_args.user_domain).id
        except exceptions.CommandError:
            if not validate_actor_existence:
                return parsed_args.user
            raise

    def _find_group():
        try:
            return common.find_group(identity_client_manager, parsed_args.group, parsed_args.group_domain).id
        except exceptions.CommandError:
            if not validate_actor_existence:
                return parsed_args.group
            raise
    kwargs = {}
    if parsed_args.user and parsed_args.system:
        kwargs['user'] = _find_user()
        kwargs['system'] = parsed_args.system
    elif parsed_args.user and parsed_args.domain:
        kwargs['user'] = _find_user()
        kwargs['domain'] = common.find_domain(identity_client_manager, parsed_args.domain).id
    elif parsed_args.user and parsed_args.project:
        kwargs['user'] = _find_user()
        kwargs['project'] = common.find_project(identity_client_manager, parsed_args.project, parsed_args.project_domain).id
    elif parsed_args.group and parsed_args.system:
        kwargs['group'] = _find_group()
        kwargs['system'] = parsed_args.system
    elif parsed_args.group and parsed_args.domain:
        kwargs['group'] = _find_group()
        kwargs['domain'] = common.find_domain(identity_client_manager, parsed_args.domain).id
    elif parsed_args.group and parsed_args.project:
        kwargs['group'] = _find_group()
        kwargs['project'] = common.find_project(identity_client_manager, parsed_args.project, parsed_args.project_domain).id
    kwargs['os_inherit_extension_inherited'] = parsed_args.inherited
    return kwargs