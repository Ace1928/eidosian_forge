import os
import string
import pytest
from .util import random_string
from keyring import errors
@pytest.fixture(autouse=True)
def _init_properties(self, request):
    self.keyring = self.init_keyring()
    self.credentials_created = set()
    request.addfinalizer(self.cleanup)