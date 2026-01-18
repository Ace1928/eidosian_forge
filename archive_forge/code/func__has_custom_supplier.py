import abc
import json
import os
from typing import NamedTuple
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
def _has_custom_supplier(self):
    return self._credential_source is None