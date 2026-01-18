import json
import os
import subprocess
import sys
import time
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _validate_revoke_response(self, response):
    self._validate_response_schema(response)
    if not response['success']:
        raise exceptions.RefreshError('Revoke failed with unsuccessful response.')