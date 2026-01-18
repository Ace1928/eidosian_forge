from novaclient.tests.functional.v2.legacy import test_quotas
class TestQuotasNovaClient2_36(TestQuotasNovaClient2_35):
    """Nova quotas functional tests."""
    COMPUTE_API_VERSION = '2.36'
    _quota_resources = ['instances', 'cores', 'ram', 'metadata_items', 'injected_files', 'injected_file_content_bytes', 'injected_file_path_bytes', 'key_pairs', 'server_groups', 'server_group_members']