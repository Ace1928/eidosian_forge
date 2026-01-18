from tempest.lib import exceptions
from novaclient.tests.functional import base
class TestQuotasNovaClient2_50(TestQuotaClassesNovaClient):
    """Nova quota classes functional tests for the v2.50 microversion."""
    COMPUTE_API_VERSION = '2.50'
    _included_resources = ['instances', 'cores', 'ram', 'metadata_items', 'injected_files', 'injected_file_content_bytes', 'injected_file_path_bytes', 'key_pairs', 'server_groups', 'server_group_members']
    _excluded_resources = ['floating_ips', 'fixed_ips', 'security_groups', 'security_group_rules']
    _extra_update_resources = []
    _blocked_update_resources = _excluded_resources