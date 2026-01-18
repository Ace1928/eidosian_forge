from os_client_config.tests import base
class TestImportVendors(base.TestCase):

    def test_get_profile(self):
        import os_client_config
        os_client_config.vendors.get_profile(profile_name='dummy')