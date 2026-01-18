import copy
import datetime
import random
from unittest import mock
import uuid
import freezegun
import http.client
from oslo_serialization import jsonutils
from pycadf import cadftaxonomy
import urllib
from urllib import parse as urlparse
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone import oauth1
from keystone.oauth1.backends import base
from keystone.tests import unit
from keystone.tests.unit.common import test_notifications
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
from keystone.tests.unit import test_v3
class FernetAuthTokenTests(AuthTokenTests, OAuthFlowTests):

    def config_overrides(self):
        super(FernetAuthTokenTests, self).config_overrides()
        self.config_fixture.config(group='token', provider='fernet')
        self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))

    def test_delete_keystone_tokens_by_consumer_id(self):
        self.skipTest('Fernet tokens are never persisted in the backend.')