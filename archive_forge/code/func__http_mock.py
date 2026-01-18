import base64
import datetime
import json
import os
import unittest
import mock
from mock import patch
from six.moves import http_client
from six.moves import urllib
from oauth2client import client
from oauth2client import client
from google_reauth import reauth
from google_reauth import errors
from google_reauth import reauth_creds
from google_reauth import _reauth_client
from google_reauth.reauth_creds import Oauth2WithReauthCredentials
def _http_mock(self, request_side_effect):
    request_mock = self.StartPatch('httplib2.Http.request')
    request_mock.side_effect = request_side_effect
    http_mock = self.StartPatch('httplib2.Http')
    http_mock.request = request_mock
    return http_mock