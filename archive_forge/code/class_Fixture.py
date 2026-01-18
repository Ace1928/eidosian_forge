import fixtures
from requests_mock import mocker
class Fixture(fixtures.Fixture, mocker.MockerCore):

    def __init__(self, **kwargs):
        fixtures.Fixture.__init__(self)
        mocker.MockerCore.__init__(self, **kwargs)

    def setUp(self):
        super(Fixture, self).setUp()
        self.start()
        self.addCleanup(self.stop)