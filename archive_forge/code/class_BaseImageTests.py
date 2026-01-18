from openstackclient.tests.functional import base
class BaseImageTests(base.TestCase):
    """Functional tests for Image commands"""

    @classmethod
    def setUpClass(cls):
        super(BaseImageTests, cls).setUpClass()
        cls.haz_v1_api = False