from unittest import mock
class FakeAuthenticator(mock.Mock):
    TOKEN = 'fake_token'
    ENDPOINT = 'http://www.example.com/endpoint'

    def __init__(self):
        super(FakeAuthenticator, self).__init__()
        self.get_token = mock.Mock()
        self.get_token.return_value = self.TOKEN
        self.get_endpoint = mock.Mock()
        self.get_endpoint.return_value = self.ENDPOINT