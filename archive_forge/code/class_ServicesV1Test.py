from manilaclient.tests.unit import utils
class ServicesV1Test(utils.TestCase):

    def test_import_v1_services_module(self):
        try:
            from manilaclient.v1 import services
        except Exception as e:
            msg = "module 'manilaclient.v1.services' cannot be imported with error: %s" % str(e)
            assert False, msg
        for cls in ('Service', 'ServiceManager'):
            msg = "Module 'services' has no '%s' attr." % cls
            self.assertTrue(hasattr(services, cls), msg)