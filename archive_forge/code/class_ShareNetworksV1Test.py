from manilaclient.tests.unit import utils
class ShareNetworksV1Test(utils.TestCase):

    def test_import_v1_share_networks_module(self):
        try:
            from manilaclient.v1 import share_networks
        except Exception as e:
            msg = "module 'manilaclient.v1.share_networks' cannot be imported with error: %s" % str(e)
            assert False, msg
        for cls in ('ShareNetwork', 'ShareNetworkManager'):
            msg = "Module 'share_networks' has no '%s' attr." % cls
            self.assertTrue(hasattr(share_networks, cls), msg)