from unittest import mock
import uuid
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
class AssignmentTestHelperMixin(object):
    """Mixin class to aid testing of assignments.

    This class supports data driven test plans that enable:

    - Creation of initial entities, such as domains, users, groups, projects
      and roles
    - Creation of assignments referencing the above entities
    - A set of input parameters and expected outputs to list_role_assignments
      based on the above test data

    A test plan is a dict of the form:

    test_plan = {
        entities: details and number of entities,
        group_memberships: group-user entity memberships,
        assignments: list of assignments to create,
        tests: list of pairs of input params and expected outputs}

    An example test plan:

    test_plan = {
        # First, create the entities required. Entities are specified by
        # a dict with the key being the entity type and the value an
        # entity specification which can be one of:
        #
        # - a simple number, e.g. {'users': 3} creates 3 users
        # - a dict where more information regarding the contents of the entity
        #   is required, e.g. {'domains' : {'users : 3}} creates a domain
        #   with three users
        # - a list of entity specifications if multiple are required
        #
        # The following creates a domain that contains a single user, group and
        # project, as well as creating three roles.

        'entities': {'domains': {'users': 1, 'groups': 1, 'projects': 1},
                     'roles': 3},

        # If it is required that an existing domain be used for the new
        # entities, then the id of that domain can be included in the
        # domain dict.  For example, if alternatively we wanted to add 3 users
        # to the default domain, add a second domain containing 3 projects as
        # well as 5 additional empty domains, the entities would be defined as:
        #
        # 'entities': {'domains': [{'id': DEFAULT_DOMAIN, 'users': 3},
        #                          {'projects': 3}, 5]},
        #
        # A project hierarchy can be specified within the 'projects' section by
        # nesting the 'project' key, for example to create a project with three
        # sub-projects you would use:

                     'projects': {'project': 3}

        # A more complex hierarchy can also be defined, for example the
        # following would define three projects each containing a
        # sub-project, each of which contain a further three sub-projects.

                     'projects': [{'project': {'project': 3}},
                                  {'project': {'project': 3}},
                                  {'project': {'project': 3}}]

        # If the 'roles' entity count is defined as top level key in 'entities'
        # dict then these are global roles. If it is placed within the
        # 'domain' dict, then they will be domain specific roles. A mix of
        # domain specific and global roles are allowed, with the role index
        # being calculated in the order they are defined in the 'entities'
        # dict.

        # A set of implied role specifications. In this case, prior role
        # index 0 implies role index 1, and role 1 implies roles 2 and 3.

        'roles': [{'role': 0, 'implied_roles': [1]},
                  {'role': 1, 'implied_roles': [2, 3]}]

        # A list of groups and their members. In this case make users with
        # index 0 and 1 members of group with index 0. Users and Groups are
        # indexed in the order they appear in the 'entities' key above.

        'group_memberships': [{'group': 0, 'users': [0, 1]}]

        # Next, create assignments between the entities, referencing the
        # entities by index, i.e. 'user': 0 refers to user[0]. Entities are
        # indexed in the order they appear in the 'entities' key above within
        # their entity type.

        'assignments': [{'user': 0, 'role': 0, 'domain': 0},
                        {'user': 0, 'role': 1, 'project': 0},
                        {'group': 0, 'role': 2, 'domain': 0},
                        {'user': 0, 'role': 2, 'project': 0}],

        # Finally, define an array of tests where list_role_assignment() is
        # called with the given input parameters and the results are then
        # confirmed to be as given in 'results'. Again, all entities are
        # referenced by index.

        'tests': [
            {'params': {},
             'results': [{'user': 0, 'role': 0, 'domain': 0},
                         {'user': 0, 'role': 1, 'project': 0},
                         {'group': 0, 'role': 2, 'domain': 0},
                         {'user': 0, 'role': 2, 'project': 0}]},
            {'params': {'role': 2},
             'results': [{'group': 0, 'role': 2, 'domain': 0},
                         {'user': 0, 'role': 2, 'project': 0}]}]

        # The 'params' key also supports the 'effective',
        # 'inherited_to_projects' and 'source_from_group_ids' options to
        # list_role_assignments.}

    """

    def _handle_project_spec(self, test_data, domain_id, project_spec, parent_id=None):
        """Handle the creation of a project or hierarchy of projects.

        project_spec may either be a count of the number of projects to
        create, or it may be a list of the form:

        [{'project': project_spec}, {'project': project_spec}, ...]

        This method is called recursively to handle the creation of a
        hierarchy of projects.

        """

        def _create_project(domain_id, parent_id):
            new_project = unit.new_project_ref(domain_id=domain_id, parent_id=parent_id)
            new_project = PROVIDERS.resource_api.create_project(new_project['id'], new_project)
            return new_project
        if isinstance(project_spec, list):
            for this_spec in project_spec:
                self._handle_project_spec(test_data, domain_id, this_spec, parent_id=parent_id)
        elif isinstance(project_spec, dict):
            new_proj = _create_project(domain_id, parent_id)
            test_data['projects'].append(new_proj)
            self._handle_project_spec(test_data, domain_id, project_spec['project'], parent_id=new_proj['id'])
        else:
            for _ in range(project_spec):
                test_data['projects'].append(_create_project(domain_id, parent_id))

    def _create_role(self, domain_id=None):
        new_role = unit.new_role_ref(domain_id=domain_id)
        return PROVIDERS.role_api.create_role(new_role['id'], new_role)

    def _handle_domain_spec(self, test_data, domain_spec):
        """Handle the creation of domains and their contents.

        domain_spec may either be a count of the number of empty domains to
        create, a dict describing the domain contents, or a list of
        domain_specs.

        In the case when a list is provided, this method calls itself
        recursively to handle the list elements.

        This method will insert any entities created into test_data

        """

        def _create_domain(domain_id=None):
            if domain_id is None:
                new_domain = unit.new_domain_ref()
                PROVIDERS.resource_api.create_domain(new_domain['id'], new_domain)
                return new_domain
            else:
                return PROVIDERS.resource_api.get_domain(domain_id)

        def _create_entity_in_domain(entity_type, domain_id):
            """Create a user or group entity in the domain."""
            if entity_type == 'users':
                new_entity = unit.new_user_ref(domain_id=domain_id)
                new_entity = PROVIDERS.identity_api.create_user(new_entity)
            elif entity_type == 'groups':
                new_entity = unit.new_group_ref(domain_id=domain_id)
                new_entity = PROVIDERS.identity_api.create_group(new_entity)
            elif entity_type == 'roles':
                new_entity = self._create_role(domain_id=domain_id)
            else:
                raise exception.NotImplemented()
            return new_entity
        if isinstance(domain_spec, list):
            for x in domain_spec:
                self._handle_domain_spec(test_data, x)
        elif isinstance(domain_spec, dict):
            the_domain = _create_domain(domain_spec.get('id'))
            test_data['domains'].append(the_domain)
            for entity_type, value in domain_spec.items():
                if entity_type == 'id':
                    continue
                if entity_type == 'projects':
                    self._handle_project_spec(test_data, the_domain['id'], value)
                else:
                    for _ in range(value):
                        test_data[entity_type].append(_create_entity_in_domain(entity_type, the_domain['id']))
        else:
            for _ in range(domain_spec):
                test_data['domains'].append(_create_domain())

    def create_entities(self, entity_pattern):
        """Create the entities specified in the test plan.

        Process the 'entities' key in the test plan, creating the requested
        entities. Each created entity will be added to the array of entities
        stored in the returned test_data object, e.g.:

        test_data['users'] = [user[0], user[1]....]

        """
        test_data = {}
        for entity in ['users', 'groups', 'domains', 'projects', 'roles']:
            test_data[entity] = []
        if 'domains' in entity_pattern:
            self._handle_domain_spec(test_data, entity_pattern['domains'])
        if 'roles' in entity_pattern:
            for _ in range(entity_pattern['roles']):
                test_data['roles'].append(self._create_role())
        return test_data

    def _convert_entity_shorthand(self, key, shorthand_data, reference_data):
        """Convert a shorthand entity description into a full ID reference.

        In test plan definitions, we allow a shorthand for referencing to an
        entity of the form:

        'user': 0

        which is actually shorthand for:

        'user_id': reference_data['users'][0]['id']

        This method converts the shorthand version into the full reference.

        """
        expanded_key = '%s_id' % key
        reference_index = '%ss' % key
        index_value = reference_data[reference_index][shorthand_data[key]]['id']
        return (expanded_key, index_value)

    def create_implied_roles(self, implied_pattern, test_data):
        """Create the implied roles specified in the test plan."""
        for implied_spec in implied_pattern:
            prior_role = test_data['roles'][implied_spec['role']]['id']
            if isinstance(implied_spec['implied_roles'], list):
                for this_role in implied_spec['implied_roles']:
                    implied_role = test_data['roles'][this_role]['id']
                    PROVIDERS.role_api.create_implied_role(prior_role, implied_role)
            else:
                implied_role = test_data['roles'][implied_spec['implied_roles']]['id']
                PROVIDERS.role_api.create_implied_role(prior_role, implied_role)

    def create_group_memberships(self, group_pattern, test_data):
        """Create the group memberships specified in the test plan."""
        for group_spec in group_pattern:
            group_value = test_data['groups'][group_spec['group']]['id']
            for user_index in group_spec['users']:
                user_value = test_data['users'][user_index]['id']
                PROVIDERS.identity_api.add_user_to_group(user_value, group_value)
        return test_data

    def create_assignments(self, assignment_pattern, test_data):
        """Create the assignments specified in the test plan."""
        test_data['initial_assignment_count'] = len(PROVIDERS.assignment_api.list_role_assignments())
        for assignment in assignment_pattern:
            args = {}
            for param in assignment:
                if param == 'inherited_to_projects':
                    args[param] = assignment[param]
                else:
                    key, value = self._convert_entity_shorthand(param, assignment, test_data)
                    args[key] = value
            PROVIDERS.assignment_api.create_grant(**args)
        return test_data

    def execute_assignment_cases(self, test_plan, test_data):
        """Execute the test plan, based on the created test_data."""

        def check_results(expected, actual, param_arg_count):
            if param_arg_count == 0:
                self.assertEqual(len(expected) + test_data['initial_assignment_count'], len(actual))
            else:
                self.assertThat(actual, matchers.HasLength(len(expected)))
            for each_expected in expected:
                expected_assignment = {}
                for param in each_expected:
                    if param == 'inherited_to_projects':
                        expected_assignment[param] = each_expected[param]
                    elif param == 'indirect':
                        indirect_term = {}
                        for indirect_param in each_expected[param]:
                            key, value = self._convert_entity_shorthand(indirect_param, each_expected[param], test_data)
                            indirect_term[key] = value
                        expected_assignment[param] = indirect_term
                    else:
                        key, value = self._convert_entity_shorthand(param, each_expected, test_data)
                        expected_assignment[key] = value
                self.assertIn(expected_assignment, actual)

        def convert_group_ids_sourced_from_list(index_list, reference_data):
            value_list = []
            for group_index in index_list:
                value_list.append(reference_data['groups'][group_index]['id'])
            return value_list
        for test in test_plan.get('tests', []):
            args = {}
            for param in test['params']:
                if param in ['effective', 'inherited', 'include_subtree']:
                    args[param] = test['params'][param]
                elif param == 'source_from_group_ids':
                    args[param] = convert_group_ids_sourced_from_list(test['params']['source_from_group_ids'], test_data)
                else:
                    key, value = self._convert_entity_shorthand(param, test['params'], test_data)
                    args[key] = value
            results = PROVIDERS.assignment_api.list_role_assignments(**args)
            check_results(test['results'], results, len(args))

    def execute_assignment_plan(self, test_plan):
        """Create entities, assignments and execute the test plan.

        The standard method to call to create entities and assignments and
        execute the tests as specified in the test_plan. The test_data
        dict is returned so that, if required, the caller can execute
        additional manual tests with the entities and assignments created.

        """
        test_data = self.create_entities(test_plan['entities'])
        if 'implied_roles' in test_plan:
            self.create_implied_roles(test_plan['implied_roles'], test_data)
        if 'group_memberships' in test_plan:
            self.create_group_memberships(test_plan['group_memberships'], test_data)
        if 'assignments' in test_plan:
            test_data = self.create_assignments(test_plan['assignments'], test_data)
        self.execute_assignment_cases(test_plan, test_data)
        return test_data