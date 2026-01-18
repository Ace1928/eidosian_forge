import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
def _set_expected_languages(self, all_locales=None, avail_locales=None):
    if all_locales is None:
        all_locales = []

    def fake_gettext_find(lang_id, *args, **kwargs):
        found_ret = '/glance/%s/LC_MESSAGES/glance.mo' % lang_id
        if avail_locales is None:
            return found_ret
        languages = kwargs['languages']
        if languages[0] in avail_locales:
            return found_ret
        return None
    self.mock_object(gettext, 'find', fake_gettext_find)