import json
import os
from google.auth import _helpers
import google.auth.transport.requests
import google.auth.transport.urllib3
import pytest
import requests
import urllib3
def _verify_refresh(credentials):
    if credentials.requires_scopes:
        credentials = credentials.with_scopes(['email', 'profile'])
    credentials.refresh(http_request)
    assert credentials.token
    assert credentials.valid