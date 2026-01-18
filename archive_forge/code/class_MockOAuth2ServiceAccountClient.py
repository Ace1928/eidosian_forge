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
class MockOAuth2ServiceAccountClient(oauth2_client.OAuth2ServiceAccountClient):
    """Mock service account client for testing OAuth2 with service accounts."""

    def __init__(self, client_id, private_key, password, auth_uri, token_uri, datetime_strategy):
        super(MockOAuth2ServiceAccountClient, self).__init__(client_id, private_key, password, auth_uri=auth_uri, token_uri=token_uri, datetime_strategy=datetime_strategy, ca_certs_file=DEFAULT_CA_CERTS_FILE)
        self.Reset()

    def Reset(self):
        self.fetched_token = False

    def FetchAccessToken(self, rapt_token=None):
        self.fetched_token = True
        return oauth2_client.AccessToken(ACCESS_TOKEN, GetExpiry(self.datetime_strategy, 3600), datetime_strategy=self.datetime_strategy, rapt_token=None)