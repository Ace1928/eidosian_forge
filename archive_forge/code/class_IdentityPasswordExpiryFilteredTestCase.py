import datetime
import freezegun
import http.client
from oslo_config import fixture as config_fixture
from oslo_serialization import jsonutils
from keystone.common import provider_api
import keystone.conf
from keystone.tests import unit
from keystone.tests.unit import filtering
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class IdentityPasswordExpiryFilteredTestCase(filtering.FilterTests, test_v3.RestfulTestCase):
    """Test password expiring filter on the v3 Identity API."""

    def setUp(self):
        """Setup for Identity Filter Test Cases."""
        self.config_fixture = self.useFixture(config_fixture.Config(CONF))
        super(IdentityPasswordExpiryFilteredTestCase, self).setUp()

    def load_sample_data(self):
        """Create sample data for password expiry tests.

        The test environment will consist of a single domain, containing
        a single project. It will create three users and one group.
        Each user is going to be given a role assignment on the project
        and the domain. Two of the three users are going to be placed into
        the group, which won't have any role assignments to either the
        project or the domain.

        """
        self._populate_default_domain()
        self.domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(self.domain['id'], self.domain)
        self.domain_id = self.domain['id']
        self.project = unit.new_project_ref(domain_id=self.domain_id)
        self.project_id = self.project['id']
        self.project = PROVIDERS.resource_api.create_project(self.project_id, self.project)
        self.group = unit.new_group_ref(domain_id=self.domain_id)
        self.group = PROVIDERS.identity_api.create_group(self.group)
        self.group_id = self.group['id']
        self.starttime = datetime.datetime.utcnow()
        with freezegun.freeze_time(self.starttime):
            self.config_fixture.config(group='security_compliance', password_expires_days=1)
            self.user = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
            self.config_fixture.config(group='security_compliance', password_expires_days=2)
            self.user2 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
            self.config_fixture.config(group='security_compliance', password_expires_days=3)
            self.user3 = unit.create_user(PROVIDERS.identity_api, domain_id=self.domain_id)
        self.role = unit.new_role_ref(name='admin')
        PROVIDERS.role_api.create_role(self.role['id'], self.role)
        self.role_id = self.role['id']
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user['id'], domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user2['id'], domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user3['id'], domain_id=self.domain_id)
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user['id'], project_id=self.project_id)
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user2['id'], project_id=self.project_id)
        PROVIDERS.assignment_api.create_grant(self.role_id, user_id=self.user3['id'], project_id=self.project_id)
        PROVIDERS.identity_api.add_user_to_group(self.user2['id'], self.group_id)
        PROVIDERS.identity_api.add_user_to_group(self.user3['id'], self.group_id)

    def _list_users_by_password_expires_at(self, time, operator=None):
        """Call `list_users` with `password_expires_at` filter.

        GET /users?password_expires_at={operator}:{timestamp}

        """
        url = '/users?password_expires_at='
        if operator:
            url += operator + ':'
        url += str(time)
        return url

    def _list_users_by_multiple_password_expires_at(self, first_time, first_operator, second_time, second_operator):
        """Call `list_users` with two `password_expires_at` filters.

        GET /users?password_expires_at={operator}:{timestamp}&
        {operator}:{timestamp}

        """
        url = '/users?password_expires_at=%s:%s&password_expires_at=%s:%s' % (first_operator, first_time, second_operator, second_time)
        return url

    def _format_timestamp(self, timestamp):
        return timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

    def test_list_users_by_password_expires_at(self):
        """Ensure users can be filtered on no operator, eq and neq.

        GET /users?password_expires_at={timestamp}
        GET /users?password_expires_at=eq:{timestamp}

        """
        expire_at_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)))
        resp_users = self.get(expire_at_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        expire_at_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'eq')
        resp_users = self.get(expire_at_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        expire_at_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'neq')
        resp_users = self.get(expire_at_url).result.get('users')
        self.assertEqual(self.user['id'], resp_users[0]['id'])
        self.assertEqual(self.user3['id'], resp_users[1]['id'])

    def test_list_users_by_password_expires_before(self):
        """Ensure users can be filtered on lt and lte.

        GET /users?password_expires_at=lt:{timestamp}
        GET /users?password_expires_at=lte:{timestamp}

        """
        expire_before_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2, seconds=1)), 'lt')
        resp_users = self.get(expire_before_url).result.get('users')
        self.assertEqual(self.user['id'], resp_users[0]['id'])
        self.assertEqual(self.user2['id'], resp_users[1]['id'])
        expire_before_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'lte')
        resp_users = self.get(expire_before_url).result.get('users')
        self.assertEqual(self.user['id'], resp_users[0]['id'])
        self.assertEqual(self.user2['id'], resp_users[1]['id'])

    def test_list_users_by_password_expires_after(self):
        """Ensure users can be filtered on gt and gte.

        GET /users?password_expires_at=gt:{timestamp}
        GET /users?password_expires_at=gte:{timestamp}

        """
        expire_after_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2, seconds=1)), 'gt')
        resp_users = self.get(expire_after_url).result.get('users')
        self.assertEqual(self.user3['id'], resp_users[0]['id'])
        expire_after_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'gte')
        resp_users = self.get(expire_after_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        self.assertEqual(self.user3['id'], resp_users[1]['id'])

    def test_list_users_by_password_expires_interval(self):
        """Ensure users can be filtered on time intervals.

        GET /users?password_expires_at=lt:{timestamp}&gt:{timestamp}
        GET /users?password_expires_at=lte:{timestamp}&gte:{timestamp}

        Time intervals are defined by using lt or lte and gt or gte,
        where the lt/lte time is greater than the gt/gte time.

        """
        expire_interval_url = self._list_users_by_multiple_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=3)), 'lt', self._format_timestamp(self.starttime + datetime.timedelta(days=1)), 'gt')
        resp_users = self.get(expire_interval_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        expire_interval_url = self._list_users_by_multiple_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'gte', self._format_timestamp(self.starttime + datetime.timedelta(days=2, seconds=1)), 'lte')
        resp_users = self.get(expire_interval_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])

    def test_list_users_by_password_expires_with_bad_operator_fails(self):
        """Ensure an invalid operator returns a Bad Request.

        GET /users?password_expires_at={invalid_operator}:{timestamp}
        GET /users?password_expires_at={operator}:{timestamp}&
        {invalid_operator}:{timestamp}

        """
        bad_op_url = self._list_users_by_password_expires_at(self._format_timestamp(self.starttime), 'x')
        self.get(bad_op_url, expected_status=http.client.BAD_REQUEST)
        bad_op_url = self._list_users_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'lt', self._format_timestamp(self.starttime), 'x')
        self.get(bad_op_url, expected_status=http.client.BAD_REQUEST)

    def test_list_users_by_password_expires_with_bad_timestamp_fails(self):
        """Ensure an invalid timestamp returns a Bad Request.

        GET /users?password_expires_at={invalid_timestamp}
        GET /users?password_expires_at={operator}:{timestamp}&
        {operator}:{invalid_timestamp}

        """
        bad_ts_url = self._list_users_by_password_expires_at(self.starttime.strftime('%S:%M:%ST%Y-%m-%d'))
        self.get(bad_ts_url, expected_status=http.client.BAD_REQUEST)
        bad_ts_url = self._list_users_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'lt', self.starttime.strftime('%S:%M:%ST%Y-%m-%d'), 'gt')
        self.get(bad_ts_url, expected_status=http.client.BAD_REQUEST)

    def _list_users_in_group_by_password_expires_at(self, time, operator=None, expected_status=http.client.OK):
        """Call `list_users_in_group` with `password_expires_at` filter.

        GET /groups/{group_id}/users?password_expires_at=
        {operator}:{timestamp}&{operator}:{timestamp}

        """
        url = '/groups/' + self.group_id + '/users?password_expires_at='
        if operator:
            url += operator + ':'
        url += str(time)
        return url

    def _list_users_in_group_by_multiple_password_expires_at(self, first_time, first_operator, second_time, second_operator, expected_status=http.client.OK):
        """Call `list_users_in_group` with two `password_expires_at` filters.

        GET /groups/{group_id}/users?password_expires_at=
        {operator}:{timestamp}&{operator}:{timestamp}

        """
        url = '/groups/' + self.group_id + '/users?password_expires_at=%s:%s&password_expires_at=%s:%s' % (first_operator, first_time, second_operator, second_time)
        return url

    def test_list_users_in_group_by_password_expires_at(self):
        """Ensure users in a group can be filtered on no operator, eq, and neq.

        GET /groups/{groupid}/users?password_expires_at={timestamp}
        GET /groups/{groupid}/users?password_expires_at=eq:{timestamp}

        """
        expire_at_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)))
        resp_users = self.get(expire_at_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        expire_at_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'eq')
        resp_users = self.get(expire_at_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        expire_at_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'neq')
        resp_users = self.get(expire_at_url).result.get('users')
        self.assertEqual(self.user3['id'], resp_users[0]['id'])

    def test_list_users_in_group_by_password_expires_before(self):
        """Ensure users in a group can be filtered on with lt and lte.

        GET /groups/{groupid}/users?password_expires_at=lt:{timestamp}
        GET /groups/{groupid}/users?password_expires_at=lte:{timestamp}

        """
        expire_before_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2, seconds=1)), 'lt')
        resp_users = self.get(expire_before_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        expire_before_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'lte')
        resp_users = self.get(expire_before_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])

    def test_list_users_in_group_by_password_expires_after(self):
        """Ensure users in a group can be filtered on with gt and gte.

        GET /groups/{groupid}/users?password_expires_at=gt:{timestamp}
        GET /groups/{groupid}/users?password_expires_at=gte:{timestamp}

        """
        expire_after_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2, seconds=1)), 'gt')
        resp_users = self.get(expire_after_url).result.get('users')
        self.assertEqual(self.user3['id'], resp_users[0]['id'])
        expire_after_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'gte')
        resp_users = self.get(expire_after_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        self.assertEqual(self.user3['id'], resp_users[1]['id'])

    def test_list_users_in_group_by_password_expires_interval(self):
        """Ensure users in a group can be filtered on time intervals.

        GET /groups/{groupid}/users?password_expires_at=
        lt:{timestamp}&gt:{timestamp}
        GET /groups/{groupid}/users?password_expires_at=
        lte:{timestamp}&gte:{timestamp}

        Time intervals are defined by using lt or lte and gt or gte,
        where the lt/lte time is greater than the gt/gte time.

        """
        expire_interval_url = self._list_users_in_group_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'gt', self._format_timestamp(self.starttime + datetime.timedelta(days=3, seconds=1)), 'lt')
        resp_users = self.get(expire_interval_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        self.assertEqual(self.user3['id'], resp_users[1]['id'])
        expire_interval_url = self._list_users_in_group_by_multiple_password_expires_at(self._format_timestamp(self.starttime + datetime.timedelta(days=2)), 'gte', self._format_timestamp(self.starttime + datetime.timedelta(days=3)), 'lte')
        resp_users = self.get(expire_interval_url).result.get('users')
        self.assertEqual(self.user2['id'], resp_users[0]['id'])
        self.assertEqual(self.user3['id'], resp_users[1]['id'])

    def test_list_users_in_group_by_password_expires_bad_operator_fails(self):
        """Ensure an invalid operator returns a Bad Request.

        GET /groups/{groupid}/users?password_expires_at=
        {invalid_operator}:{timestamp}
        GET /groups/{group_id}/users?password_expires_at=
        {operator}:{timestamp}&{invalid_operator}:{timestamp}

        """
        bad_op_url = self._list_users_in_group_by_password_expires_at(self._format_timestamp(self.starttime), 'bad')
        self.get(bad_op_url, expected_status=http.client.BAD_REQUEST)
        bad_op_url = self._list_users_in_group_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'lt', self._format_timestamp(self.starttime), 'x')
        self.get(bad_op_url, expected_status=http.client.BAD_REQUEST)

    def test_list_users_in_group_by_password_expires_bad_timestamp_fails(self):
        """Ensure and invalid timestamp returns a Bad Request.

        GET /groups/{groupid}/users?password_expires_at={invalid_timestamp}
        GET /groups/{groupid}/users?password_expires_at={operator}:{timestamp}&
        {operator}:{invalid_timestamp}

        """
        bad_ts_url = self._list_users_in_group_by_password_expires_at(self.starttime.strftime('%S:%M:%ST%Y-%m-%d'))
        self.get(bad_ts_url, expected_status=http.client.BAD_REQUEST)
        bad_ts_url = self._list_users_in_group_by_multiple_password_expires_at(self._format_timestamp(self.starttime), 'lt', self.starttime.strftime('%S:%M:%ST%Y-%m-%d'), 'gt')
        self.get(bad_ts_url, expected_status=http.client.BAD_REQUEST)