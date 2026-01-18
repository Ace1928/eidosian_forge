from oslotest import base
class KeyTestCase(TestCase):

    def _create_key(self):
        raise NotImplementedError()

    def setUp(self):
        super(KeyTestCase, self).setUp()
        self.key = self._create_key()