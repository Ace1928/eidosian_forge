import json
from launchpadlib.credentials import Credentials
from launchpadlib.uris import lookup_web_root
class RequestTokenApp(object):
    """An application that creates request tokens."""

    def __init__(self, web_root, consumer_name, context):
        """Initialize."""
        self.web_root = lookup_web_root(web_root)
        self.credentials = Credentials(consumer_name)
        self.context = context

    def run(self):
        """Get a request token and return JSON information about it."""
        token = self.credentials.get_request_token(self.context, self.web_root, token_format=Credentials.DICT_TOKEN_FORMAT)
        return json.dumps(token)