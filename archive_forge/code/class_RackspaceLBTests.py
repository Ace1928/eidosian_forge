import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
class RackspaceLBTests(unittest.TestCase):

    def setUp(self):
        RackspaceLBDriver.connectionCls.conn_class = RackspaceLBMockHttp
        RackspaceLBMockHttp.type = None
        self.driver = RackspaceLBDriver('user', 'key')
        self.driver.connection.poll_interval = 0.0
        self.driver.connection._populate_hosts_and_request_paths()

    def test_force_auth_token_kwargs(self):
        base_url = 'https://ord.loadbalancer.api.rackspacecloud.com/v1.0/11111'
        kwargs = {'ex_force_auth_token': 'some-auth-token', 'ex_force_base_url': base_url}
        driver = RackspaceLBDriver('user', 'key', **kwargs)
        driver.list_balancers()
        self.assertEqual(kwargs['ex_force_auth_token'], driver.connection.auth_token)
        self.assertEqual('/v1.0/11111', driver.connection.request_path)

    def test_force_auth_url_kwargs(self):
        kwargs = {'ex_force_auth_version': '2.0', 'ex_force_auth_url': 'https://identity.api.rackspace.com'}
        driver = RackspaceLBDriver('user', 'key', **kwargs)
        self.assertEqual(kwargs['ex_force_auth_url'], driver.connection._ex_force_auth_url)
        self.assertEqual(kwargs['ex_force_auth_version'], driver.connection._auth_version)

    def test_gets_auth_2_0_endpoint_defaults_to_ord_region(self):
        driver = RackspaceLBDriver('user', 'key', ex_force_auth_version='2.0_password')
        driver.connection._populate_hosts_and_request_paths()
        self.assertEqual('https://ord.loadbalancers.api.rackspacecloud.com/v1.0/11111', driver.connection.get_endpoint())

    def test_gets_auth_2_0_endpoint_for_dfw(self):
        driver = RackspaceLBDriver('user', 'key', ex_force_auth_version='2.0_password', ex_force_region='dfw')
        driver.connection._populate_hosts_and_request_paths()
        self.assertEqual('https://dfw.loadbalancers.api.rackspacecloud.com/v1.0/11111', driver.connection.get_endpoint())

    def test_list_protocols(self):
        protocols = self.driver.list_protocols()
        self.assertEqual(len(protocols), 10)
        self.assertTrue('http' in protocols)

    def test_ex_list_protocols_with_default_ports(self):
        protocols = self.driver.ex_list_protocols_with_default_ports()
        self.assertEqual(len(protocols), 10)
        self.assertTrue(('http', 80) in protocols)

    def test_list_supported_algorithms(self):
        algorithms = self.driver.list_supported_algorithms()
        self.assertTrue(Algorithm.RANDOM in algorithms)
        self.assertTrue(Algorithm.ROUND_ROBIN in algorithms)
        self.assertTrue(Algorithm.LEAST_CONNECTIONS in algorithms)
        self.assertTrue(Algorithm.WEIGHTED_ROUND_ROBIN in algorithms)
        self.assertTrue(Algorithm.WEIGHTED_LEAST_CONNECTIONS in algorithms)

    def test_ex_list_algorithms(self):
        algorithms = self.driver.ex_list_algorithm_names()
        self.assertTrue('RANDOM' in algorithms)
        self.assertTrue('ROUND_ROBIN' in algorithms)
        self.assertTrue('LEAST_CONNECTIONS' in algorithms)
        self.assertTrue('WEIGHTED_ROUND_ROBIN' in algorithms)
        self.assertTrue('WEIGHTED_LEAST_CONNECTIONS' in algorithms)

    def test_list_balancers(self):
        balancers = self.driver.list_balancers()
        self.assertEqual(len(balancers), 2)
        self.assertEqual(balancers[0].name, 'test0')
        self.assertEqual(balancers[0].id, '8155')
        self.assertEqual(balancers[0].port, 80)
        self.assertEqual(balancers[0].ip, '1.1.1.25')
        self.assertTrue(balancers[0].extra.get('service_name') is not None)
        self.assertTrue(balancers[0].extra.get('uri') is not None)
        self.assertEqual(balancers[1].name, 'test1')
        self.assertEqual(balancers[1].id, '8156')

    def test_list_balancers_ex_member_address(self):
        RackspaceLBMockHttp.type = 'EX_MEMBER_ADDRESS'
        balancers = self.driver.list_balancers(ex_member_address='127.0.0.1')
        self.assertEqual(len(balancers), 3)
        self.assertEqual(balancers[0].name, 'First Loadbalancer')
        self.assertEqual(balancers[0].id, '1')
        self.assertEqual(balancers[1].name, 'Second Loadbalancer')
        self.assertEqual(balancers[1].id, '2')
        self.assertEqual(balancers[2].name, 'Third Loadbalancer')
        self.assertEqual(balancers[2].id, '8')

    def test_create_balancer(self):
        balancer = self.driver.create_balancer(name='test2', port=80, algorithm=Algorithm.ROUND_ROBIN, members=(Member(None, '10.1.0.10', 80, extra={'condition': MemberCondition.DISABLED, 'weight': 10}), Member(None, '10.1.0.11', 80)))
        self.assertEqual(balancer.name, 'test2')
        self.assertEqual(balancer.id, '8290')

    def test_ex_create_balancer(self):
        RackspaceLBDriver.connectionCls.conn_class = RackspaceLBWithVIPMockHttp
        RackspaceLBMockHttp.type = None
        driver = RackspaceLBDriver('user', 'key')
        balancer = driver.ex_create_balancer(name='test2', port=80, algorithm=Algorithm.ROUND_ROBIN, members=(Member(None, '10.1.0.11', 80),), vip='12af')
        self.assertEqual(balancer.name, 'test2')
        self.assertEqual(balancer.id, '8290')

    def test_destroy_balancer(self):
        balancer = self.driver.list_balancers()[0]
        ret = self.driver.destroy_balancer(balancer)
        self.assertTrue(ret)

    def test_ex_destroy_balancers(self):
        balancers = self.driver.list_balancers()
        ret = self.driver.ex_destroy_balancers(balancers)
        self.assertTrue(ret)

    def test_get_balancer(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        self.assertEqual(balancer.name, 'test2')
        self.assertEqual(balancer.id, '8290')

    def test_get_balancer_extra_vips(self):
        balancer = self.driver.get_balancer(balancer_id='18940')
        self.assertEqual(balancer.extra['virtualIps'], [{'address': '50.56.49.149', 'id': 2359, 'type': 'PUBLIC', 'ipVersion': 'IPV4'}])

    def test_get_balancer_extra_public_source_ipv4(self):
        balancer = self.driver.get_balancer(balancer_id='18940')
        self.assertEqual(balancer.extra['ipv4PublicSource'], '184.106.100.25')

    def test_get_balancer_extra_public_source_ipv6(self):
        balancer = self.driver.get_balancer(balancer_id='18940')
        self.assertEqual(balancer.extra['ipv6PublicSource'], '2001:4801:7901::6/64')

    def test_get_balancer_extra_private_source_ipv4(self):
        balancer = self.driver.get_balancer(balancer_id='18940')
        self.assertEqual(balancer.extra['ipv4PrivateSource'], '10.183.252.25')

    def test_get_balancer_extra_members(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.extra['members']
        self.assertEqual(3, len(members))
        self.assertEqual('10.1.0.11', members[0].ip)
        self.assertEqual('10.1.0.10', members[1].ip)
        self.assertEqual('10.1.0.9', members[2].ip)

    def test_get_balancer_extra_created(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        created_8290 = datetime.datetime(2011, 4, 7, 16, 27, 50)
        self.assertEqual(created_8290, balancer.extra['created'])

    def test_get_balancer_extra_updated(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        updated_8290 = datetime.datetime(2011, 4, 7, 16, 28, 12)
        self.assertEqual(updated_8290, balancer.extra['updated'])

    def test_get_balancer_extra_access_list(self):
        balancer = self.driver.get_balancer(balancer_id='94698')
        access_list = balancer.extra['accessList']
        self.assertEqual(3, len(access_list))
        self.assertEqual(2883, access_list[0].id)
        self.assertEqual('0.0.0.0/0', access_list[0].address)
        self.assertEqual(RackspaceAccessRuleType.DENY, access_list[0].rule_type)
        self.assertEqual(2884, access_list[1].id)
        self.assertEqual('2001:4801:7901::6/64', access_list[1].address)
        self.assertEqual(RackspaceAccessRuleType.ALLOW, access_list[1].rule_type)
        self.assertEqual(3006, access_list[2].id)
        self.assertEqual('8.8.8.8/0', access_list[2].address)
        self.assertEqual(RackspaceAccessRuleType.DENY, access_list[2].rule_type)

    def test_get_balancer_algorithm(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        self.assertEqual(balancer.extra['algorithm'], Algorithm.RANDOM)

    def test_get_balancer_protocol(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        self.assertEqual(balancer.extra['protocol'], 'HTTP')

    def test_get_balancer_weighted_round_robin_algorithm(self):
        balancer = self.driver.get_balancer(balancer_id='94692')
        self.assertEqual(balancer.extra['algorithm'], Algorithm.WEIGHTED_ROUND_ROBIN)

    def test_get_balancer_weighted_least_connections_algorithm(self):
        balancer = self.driver.get_balancer(balancer_id='94693')
        self.assertEqual(balancer.extra['algorithm'], Algorithm.WEIGHTED_LEAST_CONNECTIONS)

    def test_get_balancer_unknown_algorithm(self):
        balancer = self.driver.get_balancer(balancer_id='94694')
        self.assertFalse('algorithm' in balancer.extra)

    def test_get_balancer_connect_health_monitor(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        balancer_health_monitor = balancer.extra['healthMonitor']
        self.assertEqual(balancer_health_monitor.type, 'CONNECT')
        self.assertEqual(balancer_health_monitor.delay, 10)
        self.assertEqual(balancer_health_monitor.timeout, 5)
        self.assertEqual(balancer_health_monitor.attempts_before_deactivation, 2)

    def test_get_balancer_http_health_monitor(self):
        balancer = self.driver.get_balancer(balancer_id='94696')
        balancer_health_monitor = balancer.extra['healthMonitor']
        self.assertEqual(balancer_health_monitor.type, 'HTTP')
        self.assertEqual(balancer_health_monitor.delay, 10)
        self.assertEqual(balancer_health_monitor.timeout, 5)
        self.assertEqual(balancer_health_monitor.attempts_before_deactivation, 2)
        self.assertEqual(balancer_health_monitor.path, '/')
        self.assertEqual(balancer_health_monitor.status_regex, '^[234][0-9][0-9]$')
        self.assertEqual(balancer_health_monitor.body_regex, 'Hello World!')

    def test_get_balancer_https_health_monitor(self):
        balancer = self.driver.get_balancer(balancer_id='94697')
        balancer_health_monitor = balancer.extra['healthMonitor']
        self.assertEqual(balancer_health_monitor.type, 'HTTPS')
        self.assertEqual(balancer_health_monitor.delay, 15)
        self.assertEqual(balancer_health_monitor.timeout, 12)
        self.assertEqual(balancer_health_monitor.attempts_before_deactivation, 5)
        self.assertEqual(balancer_health_monitor.path, '/test')
        self.assertEqual(balancer_health_monitor.status_regex, '^[234][0-9][0-9]$')
        self.assertEqual(balancer_health_monitor.body_regex, 'abcdef')

    def test_get_balancer_connection_throttle(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        balancer_connection_throttle = balancer.extra['connectionThrottle']
        self.assertEqual(balancer_connection_throttle.min_connections, 50)
        self.assertEqual(balancer_connection_throttle.max_connections, 200)
        self.assertEqual(balancer_connection_throttle.max_connection_rate, 50)
        self.assertEqual(balancer_connection_throttle.rate_interval_seconds, 10)

    def test_get_session_persistence(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        self.assertEqual(balancer.extra['sessionPersistenceType'], 'HTTP_COOKIE')

    def test_get_connection_logging(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        self.assertEqual(balancer.extra['connectionLoggingEnabled'], True)

    def test_get_error_page(self):
        balancer = self.driver.get_balancer(balancer_id='18940')
        error_page = self.driver.ex_get_balancer_error_page(balancer)
        self.assertTrue('The service is temporarily unavailable' in error_page)

    def test_get_access_list(self):
        balancer = self.driver.get_balancer(balancer_id='18940')
        deny_rule, allow_rule = self.driver.ex_balancer_access_list(balancer)
        self.assertEqual(deny_rule.id, 2883)
        self.assertEqual(deny_rule.rule_type, RackspaceAccessRuleType.DENY)
        self.assertEqual(deny_rule.address, '0.0.0.0/0')
        self.assertEqual(allow_rule.id, 2884)
        self.assertEqual(allow_rule.address, '2001:4801:7901::6/64')
        self.assertEqual(allow_rule.rule_type, RackspaceAccessRuleType.ALLOW)

    def test_ex_create_balancer_access_rule(self):
        balancer = self.driver.get_balancer(balancer_id='94698')
        rule = RackspaceAccessRule(rule_type=RackspaceAccessRuleType.DENY, address='0.0.0.0/0')
        rule = self.driver.ex_create_balancer_access_rule(balancer, rule)
        self.assertEqual(2883, rule.id)

    def test_ex_create_balancer_access_rule_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94698')
        rule = RackspaceAccessRule(rule_type=RackspaceAccessRuleType.DENY, address='0.0.0.0/0')
        resp = self.driver.ex_create_balancer_access_rule_no_poll(balancer, rule)
        self.assertTrue(resp)

    def test_ex_create_balancer_access_rules(self):
        balancer = self.driver.get_balancer(balancer_id='94699')
        rules = [RackspaceAccessRule(rule_type=RackspaceAccessRuleType.ALLOW, address='2001:4801:7901::6/64'), RackspaceAccessRule(rule_type=RackspaceAccessRuleType.DENY, address='8.8.8.8/0')]
        rules = self.driver.ex_create_balancer_access_rules(balancer, rules)
        self.assertEqual(2, len(rules))
        self.assertEqual(2884, rules[0].id)
        self.assertEqual(3006, rules[1].id)

    def test_ex_create_balancer_access_rules_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94699')
        rules = [RackspaceAccessRule(rule_type=RackspaceAccessRuleType.ALLOW, address='2001:4801:7901::6/64'), RackspaceAccessRule(rule_type=RackspaceAccessRuleType.DENY, address='8.8.8.8/0')]
        resp = self.driver.ex_create_balancer_access_rules_no_poll(balancer, rules)
        self.assertTrue(resp)

    def test_ex_destroy_balancer_access_rule(self):
        balancer = self.driver.get_balancer(balancer_id='94698')
        rule = RackspaceAccessRule(id='1007', rule_type=RackspaceAccessRuleType.ALLOW, address='10.45.13.5/12')
        balancer = self.driver.ex_destroy_balancer_access_rule(balancer, rule)
        rule_ids = [r.id for r in balancer.extra['accessList']]
        self.assertTrue(1007 not in rule_ids)

    def test_ex_destroy_balancer_access_rule_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94698')
        rule = RackspaceAccessRule(id=1007, rule_type=RackspaceAccessRuleType.ALLOW, address='10.45.13.5/12')
        resp = self.driver.ex_destroy_balancer_access_rule_no_poll(balancer, rule)
        self.assertTrue(resp)

    def test_ex_destroy_balancer_access_rules(self):
        balancer = self.driver.get_balancer(balancer_id='94699')
        balancer = self.driver.ex_destroy_balancer_access_rules(balancer, balancer.extra['accessList'])
        self.assertEqual('94699', balancer.id)

    def test_ex_destroy_balancer_access_rules_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94699')
        resp = self.driver.ex_destroy_balancer_access_rules_no_poll(balancer, balancer.extra['accessList'])
        self.assertTrue(resp)

    def test_ex_update_balancer_health_monitor(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        monitor = RackspaceHealthMonitor(type='CONNECT', delay=10, timeout=5, attempts_before_deactivation=2)
        balancer = self.driver.ex_update_balancer_health_monitor(balancer, monitor)
        updated_monitor = balancer.extra['healthMonitor']
        self.assertEqual('CONNECT', updated_monitor.type)
        self.assertEqual(10, updated_monitor.delay)
        self.assertEqual(5, updated_monitor.timeout)
        self.assertEqual(2, updated_monitor.attempts_before_deactivation)

    def test_ex_update_balancer_http_health_monitor(self):
        balancer = self.driver.get_balancer(balancer_id='94696')
        monitor = RackspaceHTTPHealthMonitor(type='HTTP', delay=10, timeout=5, attempts_before_deactivation=2, path='/', status_regex='^[234][0-9][0-9]$', body_regex='Hello World!')
        balancer = self.driver.ex_update_balancer_health_monitor(balancer, monitor)
        updated_monitor = balancer.extra['healthMonitor']
        self.assertEqual('HTTP', updated_monitor.type)
        self.assertEqual(10, updated_monitor.delay)
        self.assertEqual(5, updated_monitor.timeout)
        self.assertEqual(2, updated_monitor.attempts_before_deactivation)
        self.assertEqual('/', updated_monitor.path)
        self.assertEqual('^[234][0-9][0-9]$', updated_monitor.status_regex)
        self.assertEqual('Hello World!', updated_monitor.body_regex)

    def test_ex_update_balancer_health_monitor_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        monitor = RackspaceHealthMonitor(type='CONNECT', delay=10, timeout=5, attempts_before_deactivation=2)
        resp = self.driver.ex_update_balancer_health_monitor_no_poll(balancer, monitor)
        self.assertTrue(resp)

    def test_ex_update_balancer_http_health_monitor_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94696')
        monitor = RackspaceHTTPHealthMonitor(type='HTTP', delay=10, timeout=5, attempts_before_deactivation=2, path='/', status_regex='^[234][0-9][0-9]$', body_regex='Hello World!')
        resp = self.driver.ex_update_balancer_health_monitor_no_poll(balancer, monitor)
        self.assertTrue(resp)

    def test_ex_update_balancer_http_health_monitor_with_no_option_body_regex(self):
        balancer = self.driver.get_balancer(balancer_id='94700')
        monitor = RackspaceHTTPHealthMonitor(type='HTTP', delay=10, timeout=5, attempts_before_deactivation=2, path='/', status_regex='^[234][0-9][0-9]$', body_regex='')
        balancer = self.driver.ex_update_balancer_health_monitor(balancer, monitor)
        updated_monitor = balancer.extra['healthMonitor']
        self.assertEqual('HTTP', updated_monitor.type)
        self.assertEqual(10, updated_monitor.delay)
        self.assertEqual(5, updated_monitor.timeout)
        self.assertEqual(2, updated_monitor.attempts_before_deactivation)
        self.assertEqual('/', updated_monitor.path)
        self.assertEqual('^[234][0-9][0-9]$', updated_monitor.status_regex)
        self.assertEqual('', updated_monitor.body_regex)

    def test_ex_disable_balancer_health_monitor(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        balancer = self.driver.ex_disable_balancer_health_monitor(balancer)
        self.assertTrue('healthMonitor' not in balancer.extra)

    def test_ex_disable_balancer_health_monitor_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        resp = self.driver.ex_disable_balancer_health_monitor_no_poll(balancer)
        self.assertTrue(resp)

    def test_ex_update_balancer_connection_throttle(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        connection_throttle = RackspaceConnectionThrottle(max_connections=200, min_connections=50, max_connection_rate=50, rate_interval_seconds=10)
        balancer = self.driver.ex_update_balancer_connection_throttle(balancer, connection_throttle)
        updated_throttle = balancer.extra['connectionThrottle']
        self.assertEqual(200, updated_throttle.max_connections)
        self.assertEqual(50, updated_throttle.min_connections)
        self.assertEqual(50, updated_throttle.max_connection_rate)
        self.assertEqual(10, updated_throttle.rate_interval_seconds)

    def test_ex_update_balancer_connection_throttle_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        connection_throttle = RackspaceConnectionThrottle(max_connections=200, min_connections=50, max_connection_rate=50, rate_interval_seconds=10)
        resp = self.driver.ex_update_balancer_connection_throttle_no_poll(balancer, connection_throttle)
        self.assertTrue(resp)

    def test_ex_disable_balancer_connection_throttle(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        balancer = self.driver.ex_disable_balancer_connection_throttle(balancer)
        self.assertTrue('connectionThrottle' not in balancer.extra)

    def test_ex_disable_balancer_connection_throttle_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        resp = self.driver.ex_disable_balancer_connection_throttle_no_poll(balancer)
        self.assertTrue(resp)

    def test_ex_enable_balancer_connection_logging(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        balancer = self.driver.ex_enable_balancer_connection_logging(balancer)
        self.assertTrue(balancer.extra['connectionLoggingEnabled'])

    def test_ex_enable_balancer_connection_logging_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        resp = self.driver.ex_enable_balancer_connection_logging_no_poll(balancer)
        self.assertTrue(resp)

    def test_ex_disable_balancer_connection_logging(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        balancer = self.driver.ex_disable_balancer_connection_logging(balancer)
        self.assertFalse(balancer.extra['connectionLoggingEnabled'])

    def test_ex_disable_balancer_connection_logging_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        resp = self.driver.ex_disable_balancer_connection_logging_no_poll(balancer)
        self.assertTrue(resp)

    def test_ex_enable_balancer_session_persistence(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        balancer = self.driver.ex_enable_balancer_session_persistence(balancer)
        persistence_type = balancer.extra['sessionPersistenceType']
        self.assertEqual('HTTP_COOKIE', persistence_type)

    def test_ex_enable_balancer_session_persistence_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        resp = self.driver.ex_enable_balancer_session_persistence_no_poll(balancer)
        self.assertTrue(resp)

    def test_disable_balancer_session_persistence(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        balancer = self.driver.ex_disable_balancer_session_persistence(balancer)
        self.assertTrue('sessionPersistenceType' not in balancer.extra)

    def test_disable_balancer_session_persistence_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        resp = self.driver.ex_disable_balancer_session_persistence_no_poll(balancer)
        self.assertTrue(resp)

    def test_ex_update_balancer_error_page(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        content = '<html>Generic Error Page</html>'
        balancer = self.driver.ex_update_balancer_error_page(balancer, content)
        error_page_content = self.driver.ex_get_balancer_error_page(balancer)
        self.assertEqual(content, error_page_content)

    def test_ex_update_balancer_error_page_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        content = '<html>Generic Error Page</html>'
        resp = self.driver.ex_update_balancer_error_page_no_poll(balancer, content)
        self.assertTrue(resp)

    def test_ex_disable_balancer_custom_error_page_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='94695')
        resp = self.driver.ex_disable_balancer_custom_error_page_no_poll(balancer)
        self.assertTrue(resp)

    def test_ex_disable_balancer_custom_error_page(self):
        fixtures = LoadBalancerFileFixtures('rackspace')
        error_page_fixture = json.loads(fixtures.load('error_page_default.json'))
        default_error_page = error_page_fixture['errorpage']['content']
        balancer = self.driver.get_balancer(balancer_id='94695')
        balancer = self.driver.ex_disable_balancer_custom_error_page(balancer)
        error_page_content = self.driver.ex_get_balancer_error_page(balancer)
        self.assertEqual(default_error_page, error_page_content)

    def test_balancer_list_members(self):
        expected = {'10.1.0.10:80', '10.1.0.11:80', '10.1.0.9:8080'}
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.list_members()
        self.assertEqual(len(members), 3)
        self.assertEqual(members[0].balancer, balancer)
        self.assertEqual(expected, {'{}:{}'.format(member.ip, member.port) for member in members})

    def test_balancer_members_extra_weight(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.list_members()
        self.assertEqual(12, members[0].extra['weight'])
        self.assertEqual(8, members[1].extra['weight'])

    def test_balancer_members_extra_condition(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.list_members()
        self.assertEqual(MemberCondition.ENABLED, members[0].extra['condition'])
        self.assertEqual(MemberCondition.DISABLED, members[1].extra['condition'])
        self.assertEqual(MemberCondition.DRAINING, members[2].extra['condition'])

    def test_balancer_members_extra_status(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.list_members()
        self.assertEqual('ONLINE', members[0].extra['status'])
        self.assertEqual('OFFLINE', members[1].extra['status'])
        self.assertEqual('DRAINING', members[2].extra['status'])

    def test_balancer_attach_member(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        extra = {'condition': MemberCondition.DISABLED, 'weight': 10}
        member = balancer.attach_member(Member(None, ip='10.1.0.12', port='80', extra=extra))
        self.assertEqual(member.ip, '10.1.0.12')
        self.assertEqual(member.port, 80)

    def test_balancer_attach_member_with_no_condition_specified(self):
        balancer = self.driver.get_balancer(balancer_id='8291')
        member = balancer.attach_member(Member(None, ip='10.1.0.12', port='80'))
        self.assertEqual(member.ip, '10.1.0.12')
        self.assertEqual(member.port, 80)

    def test_balancer_attach_members(self):
        balancer = self.driver.get_balancer(balancer_id='8292')
        members = [Member(None, ip='10.1.0.12', port='80'), Member(None, ip='10.1.0.13', port='80')]
        attached_members = self.driver.ex_balancer_attach_members(balancer, members)
        first_member = attached_members[0]
        second_member = attached_members[1]
        self.assertEqual(first_member.ip, '10.1.0.12')
        self.assertEqual(first_member.port, 80)
        self.assertEqual(second_member.ip, '10.1.0.13')
        self.assertEqual(second_member.port, 80)

    def test_balancer_detach_member(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        member = balancer.list_members()[0]
        ret = balancer.detach_member(member)
        self.assertTrue(ret)

    def test_ex_detach_members(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.list_members()
        balancer = self.driver.ex_balancer_detach_members(balancer, members)
        self.assertEqual('8290', balancer.id)

    def test_ex_detach_members_no_poll(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = balancer.list_members()
        ret = self.driver.ex_balancer_detach_members_no_poll(balancer, members)
        self.assertTrue(ret)

    def test_update_balancer_protocol(self):
        balancer = LoadBalancer(id='3130', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, protocol='HTTPS')
        self.assertEqual('HTTPS', updated_balancer.extra['protocol'])

    def test_update_balancer_protocol_to_imapv2(self):
        balancer = LoadBalancer(id='3135', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, protocol='imapv2')
        self.assertEqual('IMAPv2', updated_balancer.extra['protocol'])

    def test_update_balancer_protocol_to_imapv3(self):
        balancer = LoadBalancer(id='3136', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, protocol='IMAPV3')
        self.assertEqual('IMAPv3', updated_balancer.extra['protocol'])

    def test_update_balancer_protocol_to_imapv4(self):
        balancer = LoadBalancer(id='3137', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, protocol='IMAPv4')
        self.assertEqual('IMAPv4', updated_balancer.extra['protocol'])

    def test_update_balancer_port(self):
        balancer = LoadBalancer(id='3131', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, port=1337)
        self.assertEqual(1337, updated_balancer.port)

    def test_update_balancer_name(self):
        balancer = LoadBalancer(id='3132', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, name='new_lb_name')
        self.assertEqual('new_lb_name', updated_balancer.name)

    def test_update_balancer_algorithm(self):
        balancer = LoadBalancer(id='3133', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        updated_balancer = self.driver.update_balancer(balancer, algorithm=Algorithm.ROUND_ROBIN)
        self.assertEqual(Algorithm.ROUND_ROBIN, updated_balancer.extra['algorithm'])

    def test_update_balancer_bad_algorithm_exception(self):
        balancer = LoadBalancer(id='3134', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        try:
            self.driver.update_balancer(balancer, algorithm='HAVE_MERCY_ON_OUR_SERVERS')
        except LibcloudError:
            pass
        else:
            self.fail('Should have thrown an exception with bad algorithm value')

    def test_ex_update_balancer_no_poll_protocol(self):
        balancer = LoadBalancer(id='3130', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        action_succeeded = self.driver.ex_update_balancer_no_poll(balancer, protocol='HTTPS')
        self.assertTrue(action_succeeded)

    def test_ex_update_balancer_no_poll_port(self):
        balancer = LoadBalancer(id='3131', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        action_succeeded = self.driver.ex_update_balancer_no_poll(balancer, port=1337)
        self.assertTrue(action_succeeded)

    def test_ex_update_balancer_no_poll_name(self):
        balancer = LoadBalancer(id='3132', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        action_succeeded = self.driver.ex_update_balancer_no_poll(balancer, name='new_lb_name')
        self.assertTrue(action_succeeded)

    def test_ex_update_balancer_no_poll_algorithm(self):
        balancer = LoadBalancer(id='3133', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        action_succeeded = self.driver.ex_update_balancer_no_poll(balancer, algorithm=Algorithm.ROUND_ROBIN)
        self.assertTrue(action_succeeded)

    def test_ex_update_balancer_no_poll_bad_algorithm_exception(self):
        balancer = LoadBalancer(id='3134', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
        try:
            self.driver.update_balancer(balancer, algorithm='HAVE_MERCY_ON_OUR_SERVERS')
        except LibcloudError:
            pass
        else:
            self.fail('Should have thrown exception with bad algorithm value')

    def test_ex_update_balancer_member_extra_attributes(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = self.driver.balancer_list_members(balancer)
        first_member = members[0]
        member = self.driver.ex_balancer_update_member(balancer, first_member, condition=MemberCondition.ENABLED, weight=12)
        self.assertEqual(MemberCondition.ENABLED, member.extra['condition'])
        self.assertEqual(12, member.extra['weight'])

    def test_ex_update_balancer_member_no_poll_extra_attributes(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        members = self.driver.balancer_list_members(balancer)
        first_member = members[0]
        resp = self.driver.ex_balancer_update_member_no_poll(balancer, first_member, condition=MemberCondition.ENABLED, weight=12)
        self.assertTrue(resp)

    def test_ex_list_current_usage(self):
        balancer = self.driver.get_balancer(balancer_id='8290')
        usage = self.driver.ex_list_current_usage(balancer=balancer)
        self.assertEqual(usage['loadBalancerUsageRecords'][0]['incomingTransferSsl'], 6182163)