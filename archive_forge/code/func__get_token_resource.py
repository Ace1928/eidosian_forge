from keystoneclient import exceptions as identity_exc
from keystoneclient.v3 import domains
from keystoneclient.v3 import groups
from keystoneclient.v3 import projects
from keystoneclient.v3 import users
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
def _get_token_resource(client, resource, parsed_name, parsed_domain=None):
    """Peek into the user's auth token to get resource IDs

    Look into a user's token to try and find the ID of a domain, project or
    user, when given the name. Typically non-admin users will interact with
    the CLI using names. However, by default, keystone does not allow look up
    by name since it would involve listing all entities. Instead opt to use
    the correct ID (from the token) instead.
    :param client: An identity client
    :param resource: A resource to look at in the token, this may be `domain`,
                     `project_domain`, `user_domain`, `project`, or `user`.
    :param parsed_name: This is input from parsed_args that the user is hoping
                        to find in the token.
    :param parsed_domain: This is domain filter from parsed_args that used to
                          filter the results.

    :returns: The ID of the resource from the token, or the original value from
              parsed_args if it does not match.
    """
    try:
        token = client.auth.client.get_token()
        token_data = client.tokens.get_token_data(token)
        token_dict = token_data['token']
        if resource == 'domain':
            token_dict = token_dict['project']
        obj = token_dict[resource]
        if parsed_domain and parsed_domain not in obj['domain'].values():
            return parsed_name
        if isinstance(obj, list):
            for item in obj:
                if item['name'] == parsed_name:
                    return item['id']
                if item['id'] == parsed_name:
                    return parsed_name
            return parsed_name
        return obj['id'] if obj['name'] == parsed_name else parsed_name
    except Exception:
        return parsed_name