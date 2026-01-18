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
def CreateMockUserAccountClient(mock_datetime):
    return MockOAuth2UserAccountClient(TOKEN_URI, 'clid', 'clsecret', 'ref_token_abc123', AUTH_URI, mock_datetime)