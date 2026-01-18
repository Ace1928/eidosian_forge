from .. import errors, mail_client, osutils, tests, urlutils
class DummyMailClient:

    def compose_merge_request(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs