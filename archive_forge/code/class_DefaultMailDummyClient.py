from .. import errors, mail_client, osutils, tests, urlutils
class DefaultMailDummyClient(mail_client.DefaultMail):

    def __init__(self):
        self.client = DummyMailClient()

    def _mail_client(self):
        return self.client