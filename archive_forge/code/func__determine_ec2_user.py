import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
def _determine_ec2_user(parsed_args, client_manager):
    """Determine a user several different ways.

    Assumes parsed_args has user and user_domain arguments. Attempts to find
    the user if domain scoping is provided, otherwise revert to a basic user
    call. Lastly use the currently authenticated user.

    """
    user_domain = None
    if parsed_args.user_domain:
        user_domain = common.find_domain(client_manager.identity, parsed_args.user_domain)
    if parsed_args.user:
        if user_domain is not None:
            user = utils.find_resource(client_manager.identity.users, parsed_args.user, domain_id=user_domain.id).id
        else:
            user = utils.find_resource(client_manager.identity.users, parsed_args.user).id
    else:
        user = client_manager.auth_ref.user_id
    return user