from __future__ import absolute_import
import datetime
import logging
import os
import stat
import sys
import unittest
from freezegun import freeze_time
from gcs_oauth2_boto_plugin import oauth2_client
import httplib2
def CreateMockServiceAccountClient(mock_datetime):
    return MockOAuth2ServiceAccountClient('clid', 'private_key', 'password', AUTH_URI, TOKEN_URI, mock_datetime)