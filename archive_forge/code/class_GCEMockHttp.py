import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
class GCEMockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('gce')
    json_hdr = {'content-type': 'application/json; charset=UTF-8'}

    def _get_method_name(self, type, use_param, qs, path):
        api_path = '/compute/%s' % API_VERSION
        project_path = '/projects/%s' % GCE_KEYWORD_PARAMS['project']
        path = path.replace(api_path, '')
        path = path.replace(project_path, '')
        if not path:
            path = '/project'
        method_name = super()._get_method_name(type, use_param, qs, path)
        return method_name

    def _setUsageExportBucket(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('setUsageExportBucket_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_custom_node(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_instances_custom_node.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_setMachineType(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_instances_node_name_setMachineType.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_setMachineType_notstopped(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_operations_operation_setMachineType_notstopped.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_custom_node_setMachineType(self, method, url, body, header):
        body = {'error': {'errors': [{'domain': 'global', 'reason': 'invalid', 'message': "Invalid value for field 'resource.machineTypes': 'projects/project_name/zones/us-central1-a/machineTypes/custom-1-61440'.  Resource was not found."}], 'code': 400, 'message': "Invalid value for field 'resource.machineTypes': 'projects/project_name/zones/us-central1-a/machineTypes/custom-1-61440'.  Resource was not found."}}
        return (httplib.BAD_REQUEST, body, self.json_hdr, httplib.responses[httplib.BAD_REQUEST])

    def _zones_us_central1_a_instances_stopped_node_setMachineType(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_instances_stopped_node_setMachineType.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_setMachineType(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_operations_operation_setMachineType.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_startnode(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_operations_operation_startnode.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_stopped_node_start(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_instances_stopped_node_start.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_stopped_node_stop(self, method, url, body, header):
        body = self.fixtures.load('zones_us_central1_a_instances_stopped_node_stop.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_stopped_node(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instances_stopped_node.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_stopnode(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_operations_operation_stopnode.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_stop(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instances_node_name_stop.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_acceleratorTypes_nvidia_tesla_k80(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_acceleratorTypes_nvidia_tesla_k80.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_setMetadata(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instances_node_name_setMetadata_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_setLabels(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instances_node_name_setLabels_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_images_custom_image_setLabels(self, method, url, body, headers):
        self.assertTrue('global/images/custom-image/setLabels' in url)
        body = self.fixtures.load('global_custom_image_setLabels_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _setCommonInstanceMetadata(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('setCommonInstanceMetadata_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_subnetworks(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_subnetworks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_addresses(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_addresses.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_diskTypes(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_disktypes.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_disks(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_disks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_forwardingRules(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_forwardingRules.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_instances(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_instances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_instanceGroupManagers(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_instanceGroupManagers.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_machineTypes(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_machineTypes.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_targetInstances(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_targetInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_targetPools(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_targetPools.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_backendServices(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_backendServices_post.json')
        else:
            backend_name = getattr(self.test, 'backendservices_mock', 'web-service')
            body = self.fixtures.load('global_backendServices-%s.json' % backend_name)
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_backendServices_no_backends(self, method, url, body, headers):
        body = self.fixtures.load('global_backendServices_no_backends.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_backendServices_web_service(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_backendServices_web_service_delete.json')
        else:
            body = self.fixtures.load('global_backendServices_web_service.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_forwardingRules(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_forwardingRules_post.json')
        else:
            body = self.fixtures.load('global_forwardingRules.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_forwardingRules_http_rule(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_forwardingRules_http_rule_delete.json')
        else:
            body = self.fixtures.load('global_forwardingRules_http_rule.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_httpHealthChecks(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_httpHealthChecks_post.json')
        else:
            body = self.fixtures.load('global_httpHealthChecks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_httpHealthChecks_default_health_check(self, method, url, body, headers):
        body = self.fixtures.load('global_httpHealthChecks_basic-check.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_httpHealthChecks_basic_check(self, method, url, body, headers):
        body = self.fixtures.load('global_httpHealthChecks_basic-check.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_httpHealthChecks_libcloud_lb_demo_healthcheck(self, method, url, body, headers):
        body = self.fixtures.load('global_httpHealthChecks_libcloud-lb-demo-healthcheck.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_httpHealthChecks_lchealthcheck(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_httpHealthChecks_lchealthcheck_delete.json')
        elif method == 'PUT':
            body = self.fixtures.load('global_httpHealthChecks_lchealthcheck_put.json')
        else:
            body = self.fixtures.load('global_httpHealthChecks_lchealthcheck.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_firewalls(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_firewalls_post.json')
        else:
            body = self.fixtures.load('global_firewalls.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_firewalls_lcfirewall(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_firewalls_lcfirewall_delete.json')
        elif method == 'PUT':
            body = self.fixtures.load('global_firewalls_lcfirewall_put.json')
        else:
            body = self.fixtures.load('global_firewalls_lcfirewall.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_firewalls_lcfirewall_egress(self, method, url, body, headers):
        body = self.fixtures.load('global_firewalls_lcfirewall-egress.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_firewalls_lcfirewall_deny(self, method, url, body, headers):
        body = self.fixtures.load('global_firewalls_lcfirewall-deny.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_images(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_images_post.json')
        elif 'maxResults' in url and 'pageToken' not in url:
            body = self.fixtures.load('global_images_paged.json')
        else:
            if 'maxResults' in url:
                self.assertIn('pageToken=token', url)
            body = self.fixtures.load('global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_images_debian_7_wheezy_v20131120(self, method, url, body, headers):
        body = self.fixtures.load('global_images_debian_7_wheezy_v20131120_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_images_debian_7_wheezy_v20131014_deprecate(self, method, url, body, headers):
        body = self.fixtures.load('global_images_debian_7_wheezy_v20131014_deprecate.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_images_family_coreos_beta(self, method, url, body, headers):
        body = self.fixtures.load('global_images_family_notfound.json')
        return (httplib.NOT_FOUND, body, self.json_hdr, httplib.responses[httplib.NOT_FOUND])

    def _global_images_family_coreos_stable(self, method, url, body, headers):
        body = self.fixtures.load('global_images_family_notfound.json')
        return (httplib.NOT_FOUND, body, self.json_hdr, httplib.responses[httplib.NOT_FOUND])

    def _global_images_family_nofamily(self, method, url, body, headers):
        body = self.fixtures.load('global_images_family_notfound.json')
        return (httplib.NOT_FOUND, body, self.json_hdr, httplib.responses[httplib.NOT_FOUND])

    def _global_routes(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_routes_post.json')
        else:
            body = self.fixtures.load('global_routes.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_networks_post.json')
        else:
            body = self.fixtures.load('global_networks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks_custom_network(self, method, url, body, headers):
        body = self.fixtures.load('global_networks_custom_network.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks_cf(self, method, url, body, headers):
        body = self.fixtures.load('global_networks_cf.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks_default(self, method, url, body, headers):
        body = self.fixtures.load('global_networks_default.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks_libcloud_demo_network(self, method, url, body, headers):
        body = self.fixtures.load('global_networks_libcloud-demo-network.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks_libcloud_demo_europe_network(self, method, url, body, headers):
        body = self.fixtures.load('global_networks_libcloud-demo-europe-network.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_routes_lcdemoroute(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_routes_lcdemoroute_delete.json')
        else:
            body = self.fixtures.load('global_routes_lcdemoroute.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_networks_lcnetwork(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_networks_lcnetwork_delete.json')
        else:
            body = self.fixtures.load('global_networks_lcnetwork.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_snapshots(self, method, url, body, headers):
        body = self.fixtures.load('global_snapshots.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_snapshots_lcsnapshot(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_snapshots_lcsnapshot_delete.json')
        else:
            body = self.fixtures.load('global_snapshots_lcsnapshot.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_setUsageExportBucket(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_setUsageExportBucket.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_setCommonInstanceMetadata(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_setCommonInstanceMetadata.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_backendServices_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_backendServices_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_backendServices_web_service_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_backendServices_web_service_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_forwardingRules_http_rule_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_forwardingRules_http_rule_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_forwardingRules_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_forwardingRules_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_httpHealthChecks_lchealthcheck_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_httpHealthChecks_lchealthcheck_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_images_debian7_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_images_debian7_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_httpHealthChecks_lchealthcheck_put(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_httpHealthChecks_lchealthcheck_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_httpHealthChecks_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_httpHealthChecks_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_firewalls_lcfirewall_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_firewalls_lcfirewall_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_firewalls_lcfirewall_put(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_firewalls_lcfirewall_put.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_firewalls_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_firewalls_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_routes_lcdemoroute_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_routes_lcdemoroute_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_networks_lcnetwork_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_networks_lcnetwork_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_routes_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_routes_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_networks_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_networks_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_snapshots_lcsnapshot_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_snapshots_lcsnapshot_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_image_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_image_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_addresses_lcaddressglobal_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_addresses_lcaddressglobal_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_targetHttpProxies_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_targetHttpProxies_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_targetHttpProxies_web_proxy_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_targetHttpProxies_web_proxy_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_urlMaps_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_urlMaps_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_urlMaps_web_map_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_urlMaps_web_map_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_targetHttpProxies(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_targetHttpProxies_post.json')
        else:
            body = self.fixtures.load('global_targetHttpProxies.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_targetHttpProxies_web_proxy(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_targetHttpProxies_web_proxy_delete.json')
        else:
            body = self.fixtures.load('global_targetHttpProxies_web_proxy.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_urlMaps(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_urlMaps_post.json')
        else:
            body = self.fixtures.load('global_urlMaps.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_urlMaps_web_map(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_urlMaps_web_map_delete.json')
        else:
            body = self.fixtures.load('global_urlMaps_web_map.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_east1_subnetworks_cf_972cf02e6ad49113(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-east1_subnetworks_cf_972cf02e6ad49113.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_subnetworks_cf_972cf02e6ad49112(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_subnetworks_cf_972cf02e6ad49112.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_other_name_regions_us_central1(self, method, url, body, headers):
        body = self.fixtures.load('projects_other_name_regions_us-central1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_other_name_global_networks_lcnetwork(self, method, url, body, headers):
        body = self.fixtures.load('projects_other_name_global_networks_lcnetwork.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_other_name_global_networks_cf(self, method, url, body, headers):
        body = self.fixtures.load('projects_other_name_global_networks_cf.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_other_name_global_networks_shared_network_for_mig(self, method, url, body, headers):
        body = self.fixtures.load('projects_other_name_global_networks_shared_network_for_mig.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_other_name_regions_us_central1_subnetworks_cf_972cf02e6ad49114(self, method, url, body, headers):
        body = self.fixtures.load('projects_other_name_regions_us-central1_subnetworks_cf_972cf02e6ad49114.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_other_name_regions_us_central1_subnetworks_shared_subnetwork_for_mig(self, method, url, body, headers):
        body = self.fixtures.load('projects_other_name_regions_us-central1_subnetworks_shared_subnetwork_for_mig.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_addresses_lcaddress_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_addresses_lcaddress_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_addresses_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_addresses_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_addresses_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_addresses_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_subnetworks_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_subnetworks_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_forwardingRules_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_forwardingRules_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_forwardingRules_lcforwardingrule_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_forwardingRules_lcforwardingrule_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_deleteAccessConfig(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node_name_deleteAccessConfig_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_serialPort(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instances_node_name_getSerialOutput.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_addAccessConfig(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node_name_addAccessConfig_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_setMetadata_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us_central1_a_node_name_setMetadata_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_setLabels_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us_central1_a_node_name_setLabels_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_setImageLabels_post(self, method, url, body, headers):
        body = self.fixtures.load('global_operations_operation_setImageLabels_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_targetInstances_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_targetInstances_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_addAccessConfig_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node_name_addAccessConfig_done.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_deleteAccessConfig_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node_name_deleteAccessConfig_done.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_targetInstances_lctargetinstance_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_targetInstances_lctargetinstance_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_lctargetpool_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_lctargetpool_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_lctargetpool_removeHealthCheck_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_lctargetpool_removeHealthCheck_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_lctargetpool_addHealthCheck_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_lctargetpool_addHealthCheck_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_lctargetpool_removeInstance_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_lctargetpool_removeInstance_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_lb_pool_setBackup_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_lb_pool_setBackup_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_operations_operation_regions_us_central1_targetPools_lctargetpool_addInstance_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_regions_us-central1_targetPools_lctargetpool_addInstance_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_disks_lcdisk_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_disks_lcdisk_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_setDiskAutoDelete(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instances_node_name_setDiskAutoDelete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_volume_auto_delete(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_operations_operation_volume_auto_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_disks_lcdisk_createSnapshot_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_disks_lcdisk_createSnapshot_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_disks_lcdisk_resize_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_disks_lcdisk_resize_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_disks_lcdisk_setLabels_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_disks_lcdisk_setLabels_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_disks_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_disks_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_lcnode_000_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_lcnode-000_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_lcnode_001_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_lcnode-001_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_delete(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node-name_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_attachDisk_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node-name_attachDisk_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_detachDisk_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node-name_detachDisk_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_setTags_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node-name_setTags_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_node_name_reset_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_node-name_reset_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_operations_operation_zones_europe_west1_a_instances_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_europe-west1-a_instances_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instances_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_zones_us-central1-a_instances_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _project(self, method, url, body, headers):
        body = self.fixtures.load('project.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_windows_cloud_global_licenses_windows_server_2008_r2_dc(self, method, url, body, headers):
        body = self.fixtures.load('projects_windows-cloud_global_licenses_windows_server_2008_r2_dc.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_suse_cloud_global_licenses_sles_11(self, method, url, body, headers):
        body = self.fixtures.load('projects_suse-cloud_global_licenses_sles_11.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_rhel_cloud_global_licenses_rhel_7_server(self, method, url, body, headers):
        body = self.fixtures.load('projects_rhel-cloud_global_licenses_rhel_server.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_coreos_cloud_global_licenses_coreos_stable(self, method, url, body, headers):
        body = self.fixtures.load('projects_coreos-cloud_global_licenses_coreos_stable.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_suse_cloud_global_licenses_sles_12(self, method, url, body, headers):
        body = self.fixtures.load('projects_suse-cloud_global_licenses_sles_12.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_windows_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_windows-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_windows_sql_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_windows-sql-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_rhel_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_rhel-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_coreos_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_coreos-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_coreos_cloud_global_images_family_coreos_beta(self, method, url, body, header):
        body = self.fixtures.load('projects_coreos-cloud_global_images_family_coreos_beta.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_coreos_cloud_global_images_family_coreos_stable(self, method, url, body, header):
        body = self.fixtures.load('projects_coreos-cloud_global_images_family_coreos_stable.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_cos_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_cos-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_opensuse_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_opensuse-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_ubuntu_os_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_ubuntu-os-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_centos_cloud_global_images(self, method, url, body, header):
        body = self.fixtures.load('projects_centos-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_suse_cloud_global_images(self, method, url, body, headers):
        body = self.fixtures.load('projects_suse-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_suse_byos_cloud_global_images(self, method, url, body, headers):
        body = self.fixtures.load('projects_suse-byos-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_suse_sap_cloud_global_images(self, method, url, body, headers):
        body = self.fixtures.load('projects_suse-sap-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _projects_debian_cloud_global_images(self, method, url, body, headers):
        body = self.fixtures.load('projects_debian-cloud_global_images.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions(self, method, url, body, headers):
        if 'pageToken' in url or 'filter' in url:
            body = self.fixtures.load('regions-paged-2.json')
        elif 'maxResults' in url:
            body = self.fixtures.load('regions-paged-1.json')
        else:
            body = self.fixtures.load('regions.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_addresses(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_addresses_post.json')
        else:
            body = self.fixtures.load('global_addresses.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_europe_west1(self, method, url, body, headers):
        body = self.fixtures.load('regions_europe-west1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_asia_east1(self, method, url, body, headers):
        body = self.fixtures.load('regions_asia-east1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_east1(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-east1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_subnetworks(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('regions_us-central1_subnetworks_post.json')
        else:
            body = self.fixtures.load('regions_us-central1_subnetworks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_addresses(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('regions_us-central1_addresses_post.json')
        else:
            body = self.fixtures.load('regions_us-central1_addresses.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_addresses_lcaddressglobal(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('global_addresses_lcaddressglobal_delete.json')
        else:
            body = self.fixtures.load('global_addresses_lcaddressglobal.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_addresses_lcaddress(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('regions_us-central1_addresses_lcaddress_delete.json')
        else:
            body = self.fixtures.load('regions_us-central1_addresses_lcaddress.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_addresses_testaddress(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_addresses_testaddress.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_subnetworks_subnet_1(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_subnetworks_subnet_1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_addresses_lcaddressinternal(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_addresses_lcaddressinternal.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_forwardingRules(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('regions_us-central1_forwardingRules_post.json')
        else:
            body = self.fixtures.load('regions_us-central1_forwardingRules.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_forwardingRules_libcloud_lb_demo_lb(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_forwardingRules_libcloud-lb-demo-lb.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_forwardingRules_lcforwardingrule(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('regions_us-central1_forwardingRules_lcforwardingrule_delete.json')
        else:
            body = self.fixtures.load('regions_us-central1_forwardingRules_lcforwardingrule.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_targetInstances(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_us-central1-a_targetInstances_post.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_targetInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('regions_us-central1_targetPools_post.json')
        else:
            body = self.fixtures.load('regions_us-central1_targetPools.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_targetInstances_lctargetinstance(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('zones_us-central1-a_targetInstances_lctargetinstance_delete.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_targetInstances_lctargetinstance.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lb_pool_getHealth(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lb_pool_getHealth.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lb_pool(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lb_pool.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lctargetpool(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool_delete.json')
        else:
            body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lctargetpool_sticky(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool_sticky.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_backup_pool(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_backup_pool.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_libcloud_lb_demo_lb_tp(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_libcloud-lb-demo-lb-tp.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lctargetpool_removeHealthCheck(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool_removeHealthCheck_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lctargetpool_addHealthCheck(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool_addHealthCheck_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lctargetpool_removeInstance(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool_removeInstance_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lb_pool_setBackup(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lb_pool_setBackup_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _regions_us_central1_targetPools_lctargetpool_addInstance(self, method, url, body, headers):
        body = self.fixtures.load('regions_us-central1_targetPools_lctargetpool_addInstance_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones(self, method, url, body, headers):
        body = self.fixtures.load('zones.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_asia_east_1a(self, method, url, body, headers):
        body = self.fixtures.load('zones_asia-east1-a.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_asia_east1_b(self, method, url, body, headers):
        body = self.fixtures.load('zones_asia-east1-b.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_east1_b(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-east1-b.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_diskTypes(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_diskTypes.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_diskTypes_pd_standard(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_diskTypes_pd_standard.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_diskTypes_pd_ssd(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_diskTypes_pd_ssd.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_us-central1-a_disks_post.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_disks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_lcdisk(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('zones_us-central1-a_disks_lcdisk_delete.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_disks_lcdisk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_lcdisk_createSnapshot(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_disks_lcdisk_createSnapshot_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_lcdisk_resize(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_disks_lcdisk_resize_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_lcdisk_setLabels(self, method, url, body, header):
        body = self.fixtures.load('zones_us-central1-a_disks_lcdisk_setLabel_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_node_name(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_lcnode_000(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_disks_lcnode_001(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_disks_libcloud_lb_demo_www_000(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_disks_libcloud_lb_demo_www_001(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_disks_libcloud_lb_demo_www_002(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central2_a_disks_libcloud_demo_boot_disk(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central2_a_disks_libcloud_demo_np_node(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central2_a_disks_libcloud_demo_multiple_nodes_000(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central2_a_disks_libcloud_demo_multiple_nodes_001(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_disks(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_us-central1-a_disks_post.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_disks.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_disks_libcloud_demo_europe_np_node(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_disks_libcloud_demo_europe_boot_disk(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_disks_libcloud_demo_europe_multiple_nodes_000(self, method, url, body, headers):
        body = self.fixtures.load('generic_disk.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_instances(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_europe-west1-a_instances_post.json')
        else:
            body = self.fixtures.load('zones_europe-west1-a_instances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instances(self, method, url, body, headers):
        if method == 'GET':
            body = '{}'
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_diskTypes_pd_standard(self, method, url, body, headers):
        body = self.fixtures.load('zones_europe-west1-a_diskTypes_pd_standard.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_us-central1-a_instances_post.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_instances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_sn_node_name(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instances_sn-node-name.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('zones_us-central1-a_instances_node-name_delete.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_instances_node-name.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_attachDisk(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instances_node-name_attachDisk_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_detachDisk(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instances_node-name_detachDisk_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_setTags(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instances_node-name_setTags_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_node_name_reset(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instances_node-name_reset_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_lcnode_000(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('zones_us-central1-a_instances_lcnode-000_delete.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_instances_lcnode-000.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instances_lcnode_001(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('zones_us-central1-a_instances_lcnode-001_delete.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_instances_lcnode-001.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instances_libcloud_lb_nopubip_001(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-b_instances_libcloud-lb-nopubip-001.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instances_libcloud_lb_demo_www_000(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-b_instances_libcloud-lb-demo-www-000.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instances_libcloud_lb_demo_www_001(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-b_instances_libcloud-lb-demo-www-001.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instances_libcloud_lb_demo_www_002(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-b_instances_libcloud-lb-demo-www-002.json')
        return (httplib.NOT_FOUND, body, self.json_hdr, httplib.responses[httplib.NOT_FOUND])

    def _zones_us_central1_a(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_machineTypes(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_machineTypes.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_europe_west1_a_machineTypes_n1_standard_1(self, method, url, body, headers):
        body = self.fixtures.load('zones_europe-west1-a_machineTypes_n1-standard-1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_machineTypes_n1_standard_1(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_machineTypes_n1-standard-1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myinstancegroup(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instanceGroup_myinstancegroup.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myinstancegroup2(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instanceGroup_myinstancegroup2.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instanceGroups_myinstancegroup(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-b_instanceGroup_myinstancegroup.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_east1_b_instanceGroups_myinstancegroup(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-east1-b_instanceGroup_myinstancegroup.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myinstancegroup_shared_network(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instanceGroup_myinstancegroup_shared_network.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroupManagers_myinstancegroup(self, method, url, body, headers):
        if method == 'PATCH':
            body = self.fixtures.load('zones_us-central1-a_operations_operation_zones_us-central1-a_instanceGroupManagers_insert_post.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_instanceGroupManagers_myinstancegroup.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroupManagers_myinstancegroup_shared_network(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_instanceGroupManagers_myinstancegroup_shared_network.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_b_instanceGroupManagers_myinstancegroup(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-b_instanceGroupManagers_myinstancegroup.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroupManagers_myinstancegroup_listManagedInstances(self, method, url, body, headers):
        body = self.fixtures.load('_zones_us_central1_a_instanceGroupManagers_myinstancegroup_listManagedInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_east1_b_instanceGroupManagers(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-east1-b_instanceGroupManagers.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroupManagers(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_us-central1-a_instanceGroupManagers_insert.json')
        else:
            body = self.fixtures.load('zones_us-central1-a_instanceGroupManagers.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroupManagers_insert_post(self, method, url, body, headers):
        body = self.fixtures.load('zones_us-central1-a_operations_operation_zones_us-central1-a_instanceGroupManagers_insert_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_instanceTemplates(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_instanceTemplates_insert.json')
        else:
            body = self.fixtures.load('global_instanceTemplates.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_instanceTemplates_my_instance_template1_insert(self, method, url, body, headers):
        """Redirects from _global_instanceTemplates"""
        body = self.fixtures.load('operations_operation_global_instanceTemplates_insert.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_instanceTemplates_my_instance_template1(self, method, url, body, headers):
        body = self.fixtures.load('global_instanceTemplates_my_instance_template1.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_instanceTemplates_my_instance_template_shared_network(self, method, url, body, headers):
        body = self.fixtures.load('global_instanceTemplates_my_instance_template_shared_network.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _aggregated_autoscalers(self, method, url, body, headers):
        body = self.fixtures.load('aggregated_autoscalers.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_sslCertificates(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('global_sslcertificates_post.json')
        else:
            body = self.fixtures.load('global_sslcertificates.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_sslCertificates_example(self, method, url, body, headers):
        body = self.fixtures.load('global_sslcertificates_example.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _global_operations_operation_global_sslcertificates_post(self, method, url, body, headers):
        body = self.fixtures.load('operations_operation_global_sslcertificates_post.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myname(self, method, url, body, headers):
        if method == 'DELETE':
            body = self.fixtures.load('zones_us_central1_a_instanceGroups_myname_delete.json')
        else:
            body = self.fixtures.load('zones_us_central1_a_instanceGroups_myname.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroups_myname_delete(self, method, url, body, headers):
        """Redirects from _zones_us_central1_a_instanceGroups_myname"""
        body = self.fixtures.load('operations_operation_zones_us_central1_a_instanceGroups_myname_delete.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups(self, method, url, body, headers):
        if method == 'POST':
            body = self.fixtures.load('zones_us_central1_a_instanceGroups_insert.json')
        else:
            body = self.fixtures.load('zones_us_central1_a_instanceGroups.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_zone_attribute_not_present(self, method, url, body, headers):
        if method == 'GET':
            body = self.fixtures.load('zones_us_central1_a_instanceGroups_zone_attribute_not_present.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroups_myname_insert(self, method, url, body, headers):
        """Redirects from _zones_us_central1_a_instanceGroups"""
        body = self.fixtures.load('operations_operation_zones_us_central1_a_instanceGroups_insert.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myname_listInstances(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instanceGroups_myname_listInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myname_addInstances(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instanceGroups_myname_addInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroups_myname_addInstances(self, method, url, body, headers):
        """Redirects from _zones_us_central1_a_instanceGroups_myname_addInstances"""
        body = self.fixtures.load('operations_operation_zones_us_central1_a_instanceGroups_myname_addInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myname_removeInstances(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instanceGroups_myname_removeInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroups_myname_removeInstances(self, method, url, body, headers):
        """Redirects from _zones_us_central1_a_instanceGroups_myname_removeInstances"""
        body = self.fixtures.load('operations_operation_zones_us_central1_a_instanceGroups_myname_removeInstances.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_instanceGroups_myname_setNamedPorts(self, method, url, body, headers):
        body = self.fixtures.load('zones_us_central1_a_instanceGroups_myname_setNamedPorts.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])

    def _zones_us_central1_a_operations_operation_zones_us_central1_a_instanceGroups_myname_setNamedPorts(self, method, url, body, headers):
        """Redirects from _zones_us_central1_a_instanceGroups_myname_setNamedPorts"""
        body = self.fixtures.load('operations_operation_zones_us_central1_a_instanceGroups_myname_setNamedPorts.json')
        return (httplib.OK, body, self.json_hdr, httplib.responses[httplib.OK])