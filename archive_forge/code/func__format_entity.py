import flask
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone import exception
from keystone.i18n import _
from keystone.server import flask as ks_flask
def _format_entity(self, entity):
    """Format an assignment entity for API response.

        The driver layer returns entities as dicts containing the ids of the
        actor (e.g. user or group), target (e.g. domain or project) and role.
        If it is an inherited role, then this is also indicated. Examples:

        For a non-inherited expanded assignment from group membership:
        {'user_id': user_id,
         'project_id': project_id,
         'role_id': role_id,
         'indirect': {'group_id': group_id}}

        or, for a project inherited role:

        {'user_id': user_id,
         'project_id': project_id,
         'role_id': role_id,
         'indirect': {'project_id': parent_id}}

        or, for a role that was implied by a prior role:

        {'user_id': user_id,
         'project_id': project_id,
         'role_id': role_id,
         'indirect': {'role_id': prior role_id}}

        It is possible to deduce if a role assignment came from group
        membership if it has both 'user_id' in the main body of the dict and
        'group_id' in the 'indirect' subdict, as well as it is possible to
        deduce if it has come from inheritance if it contains both a
        'project_id' in the main body of the dict and 'parent_id' in the
        'indirect' subdict.

        This function maps this into the format to be returned via the API,
        e.g. for the second example above:

        {
            'user': {
                {'id': user_id}
            },
            'scope': {
                'project': {
                    {'id': project_id}
                },
                'OS-INHERIT:inherited_to': 'projects'
            },
            'role': {
                {'id': role_id}
            },
            'links': {
                'assignment': '/OS-INHERIT/projects/parent_id/users/user_id/'
                              'roles/role_id/inherited_to_projects'
            }
        }

        """
    formatted_link = ''
    formatted_entity = {'links': {}}
    inherited_assignment = entity.get('inherited_to_projects')
    if 'project_id' in entity:
        if 'project_name' in entity:
            formatted_entity['scope'] = {'project': {'id': entity['project_id'], 'name': entity['project_name'], 'domain': {'id': entity['project_domain_id'], 'name': entity['project_domain_name']}}}
        else:
            formatted_entity['scope'] = {'project': {'id': entity['project_id']}}
        if 'domain_id' in entity.get('indirect', {}):
            inherited_assignment = True
            formatted_link = '/domains/%s' % entity['indirect']['domain_id']
        elif 'project_id' in entity.get('indirect', {}):
            inherited_assignment = True
            formatted_link = '/projects/%s' % entity['indirect']['project_id']
        else:
            formatted_link = '/projects/%s' % entity['project_id']
    elif 'domain_id' in entity:
        if 'domain_name' in entity:
            formatted_entity['scope'] = {'domain': {'id': entity['domain_id'], 'name': entity['domain_name']}}
        else:
            formatted_entity['scope'] = {'domain': {'id': entity['domain_id']}}
        formatted_link = '/domains/%s' % entity['domain_id']
    elif 'system' in entity:
        formatted_link = '/system'
        formatted_entity['scope'] = {'system': entity['system']}
    if 'user_id' in entity:
        if 'user_name' in entity:
            formatted_entity['user'] = {'id': entity['user_id'], 'name': entity['user_name'], 'domain': {'id': entity['user_domain_id'], 'name': entity['user_domain_name']}}
        else:
            formatted_entity['user'] = {'id': entity['user_id']}
        if 'group_id' in entity.get('indirect', {}):
            membership_url = ks_flask.base_url(path='/groups/%s/users/%s' % (entity['indirect']['group_id'], entity['user_id']))
            formatted_entity['links']['membership'] = membership_url
            formatted_link += '/groups/%s' % entity['indirect']['group_id']
        else:
            formatted_link += '/users/%s' % entity['user_id']
    elif 'group_id' in entity:
        if 'group_name' in entity:
            formatted_entity['group'] = {'id': entity['group_id'], 'name': entity['group_name'], 'domain': {'id': entity['group_domain_id'], 'name': entity['group_domain_name']}}
        else:
            formatted_entity['group'] = {'id': entity['group_id']}
        formatted_link += '/groups/%s' % entity['group_id']
    if 'role_name' in entity:
        formatted_entity['role'] = {'id': entity['role_id'], 'name': entity['role_name']}
        if 'role_domain_id' in entity and 'role_domain_name' in entity:
            formatted_entity['role'].update({'domain': {'id': entity['role_domain_id'], 'name': entity['role_domain_name']}})
    else:
        formatted_entity['role'] = {'id': entity['role_id']}
    prior_role_link = ''
    if 'role_id' in entity.get('indirect', {}):
        formatted_link += '/roles/%s' % entity['indirect']['role_id']
        prior_role_link = '/prior_role/%(prior)s/implies/%(implied)s' % {'prior': entity['role_id'], 'implied': entity['indirect']['role_id']}
    else:
        formatted_link += '/roles/%s' % entity['role_id']
    if inherited_assignment:
        formatted_entity['scope']['OS-INHERIT:inherited_to'] = 'projects'
        formatted_link = '/OS-INHERIT%s/inherited_to_projects' % formatted_link
    formatted_entity['links']['assignment'] = ks_flask.base_url(path=formatted_link)
    if prior_role_link:
        formatted_entity['links']['prior_role'] = ks_flask.base_url(path=prior_role_link)
    return formatted_entity