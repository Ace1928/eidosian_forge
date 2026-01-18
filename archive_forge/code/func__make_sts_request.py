import datetime
import io
import json
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils
def _make_sts_request(self, request):
    return self._sts_client.refresh_token(request, self._refresh_token)