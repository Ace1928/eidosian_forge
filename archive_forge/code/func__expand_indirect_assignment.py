import copy
import itertools
from oslo_log import log
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
def _expand_indirect_assignment(self, ref, user_id=None, project_id=None, subtree_ids=None, expand_groups=True):
    """Return a list of expanded role assignments.

        This methods is called for each discovered assignment that either needs
        a group assignment expanded into individual user assignments, or needs
        an inherited assignment to be applied to its children.

        In all cases, if either user_id and/or project_id is specified, then we
        filter the result on those values.

        If project_id is specified and subtree_ids is None, then this
        indicates that we are only interested in that one project. If
        subtree_ids is not None, then this is an indicator that any
        inherited assignments need to be expanded down the tree. The
        actual subtree_ids don't need to be used as a filter here, since we
        already ensured only those assignments that could affect them
        were passed to this method.

        If expand_groups is True then we expand groups out to a list of
        assignments, one for each member of that group.

        """

    def create_group_assignment(base_ref, user_id):
        """Create a group assignment from the provided ref."""
        ref = copy.deepcopy(base_ref)
        ref['user_id'] = user_id
        indirect = ref.setdefault('indirect', {})
        indirect['group_id'] = ref.pop('group_id')
        return ref

    def expand_group_assignment(ref, user_id):
        """Expand group role assignment.

            For any group role assignment on a target, it is replaced by a list
            of role assignments containing one for each user of that group on
            that target.

            An example of accepted ref is::

            {
                'group_id': group_id,
                'project_id': project_id,
                'role_id': role_id
            }

            Once expanded, it should be returned as a list of entities like the
            one below, one for each user_id in the provided group_id.

            ::

            {
                'user_id': user_id,
                'project_id': project_id,
                'role_id': role_id,
                'indirect' : {
                    'group_id': group_id
                }
            }

            Returned list will be formatted by the Controller, which will
            deduce a role assignment came from group membership if it has both
            'user_id' in the main body of the dict and 'group_id' in indirect
            subdict.

            """
        if user_id:
            return [create_group_assignment(ref, user_id=user_id)]
        try:
            users = PROVIDERS.identity_api.list_users_in_group(ref['group_id'])
        except exception.GroupNotFound:
            LOG.warning('Group %(group)s was not found but still has role assignments.', {'group': ref['group_id']})
            users = []
        return [create_group_assignment(ref, user_id=m['id']) for m in users]

    def expand_inherited_assignment(ref, user_id, project_id, subtree_ids, expand_groups):
        """Expand inherited role assignments.

            If expand_groups is True and this is a group role assignment on a
            target, replace it by a list of role assignments containing one for
            each user of that group, on every project under that target. If
            expand_groups is False, then return a group assignment on an
            inherited target.

            If this is a user role assignment on a specific target (i.e.
            project_id is specified, but subtree_ids is None) then simply
            format this as a single assignment (since we are effectively
            filtering on project_id). If however, project_id is None or
            subtree_ids is not None, then replace this one assignment with a
            list of role assignments for that user on every project under
            that target.

            An example of accepted ref is::

            {
                'group_id': group_id,
                'project_id': parent_id,
                'role_id': role_id,
                'inherited_to_projects': 'projects'
            }

            Once expanded, it should be returned as a list of entities like the
            one below, one for each user_id in the provided group_id and
            for each subproject_id in the project_id subtree.

            ::

            {
                'user_id': user_id,
                'project_id': subproject_id,
                'role_id': role_id,
                'indirect' : {
                    'group_id': group_id,
                    'project_id': parent_id
                }
            }

            Returned list will be formatted by the Controller, which will
            deduce a role assignment came from group membership if it has both
            'user_id' in the main body of the dict and 'group_id' in the
            'indirect' subdict, as well as it is possible to deduce if it has
            come from inheritance if it contains both a 'project_id' in the
            main body of the dict and 'parent_id' in the 'indirect' subdict.

            """

        def create_inherited_assignment(base_ref, project_id):
            """Create a project assignment from the provided ref.

                base_ref can either be a project or domain inherited
                assignment ref.

                """
            ref = copy.deepcopy(base_ref)
            indirect = ref.setdefault('indirect', {})
            if ref.get('project_id'):
                indirect['project_id'] = ref.pop('project_id')
            else:
                indirect['domain_id'] = ref.pop('domain_id')
            ref['project_id'] = project_id
            ref.pop('inherited_to_projects')
            return ref
        if project_id:
            project_ids = [project_id]
            if subtree_ids:
                project_ids += subtree_ids
                resource_api = PROVIDERS.resource_api
                if ref.get('project_id'):
                    if ref['project_id'] in project_ids:
                        project_ids = [x['id'] for x in resource_api.list_projects_in_subtree(ref['project_id'])]
        elif ref.get('domain_id'):
            project_ids = [x['id'] for x in PROVIDERS.resource_api.list_projects_in_domain(ref['domain_id'])]
        else:
            project_ids = [x['id'] for x in PROVIDERS.resource_api.list_projects_in_subtree(ref['project_id'])]
        new_refs = []
        if 'group_id' in ref:
            if expand_groups:
                for ref in expand_group_assignment(ref, user_id):
                    new_refs += [create_inherited_assignment(ref, proj_id) for proj_id in project_ids]
            else:
                new_refs += [create_inherited_assignment(ref, proj_id) for proj_id in project_ids]
        else:
            new_refs += [create_inherited_assignment(ref, proj_id) for proj_id in project_ids]
        return new_refs
    if ref.get('inherited_to_projects') == 'projects':
        return expand_inherited_assignment(ref, user_id, project_id, subtree_ids, expand_groups)
    elif 'group_id' in ref and expand_groups:
        return expand_group_assignment(ref, user_id)
    return [ref]