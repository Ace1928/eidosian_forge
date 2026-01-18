import datetime
import functools
import os
import sys
import freezegun
import mock
import OpenSSL
import pytest  # type: ignore
import requests
import requests.adapters
from six.moves import http_client
from google.auth import environment_vars
from google.auth import exceptions
import google.auth.credentials
import google.auth.transport._custom_tls_signer
import google.auth.transport._mtls_helper
import google.auth.transport.requests
from google.oauth2 import service_account
from tests.transport import compliance
class TestTimeoutGuard(object):

    def make_guard(self, *args, **kwargs):
        return google.auth.transport.requests.TimeoutGuard(*args, **kwargs)

    def test_tracks_elapsed_time_w_numeric_timeout(self, frozen_time):
        with self.make_guard(timeout=10) as guard:
            frozen_time.tick(delta=datetime.timedelta(seconds=3.8))
        assert guard.remaining_timeout == 6.2

    def test_tracks_elapsed_time_w_tuple_timeout(self, frozen_time):
        with self.make_guard(timeout=(16, 19)) as guard:
            frozen_time.tick(delta=datetime.timedelta(seconds=3.8))
        assert guard.remaining_timeout == (12.2, 15.2)

    def test_noop_if_no_timeout(self, frozen_time):
        with self.make_guard(timeout=None) as guard:
            frozen_time.tick(delta=datetime.timedelta(days=3650))
        assert guard.remaining_timeout is None

    def test_timeout_error_w_numeric_timeout(self, frozen_time):
        with pytest.raises(requests.exceptions.Timeout):
            with self.make_guard(timeout=10) as guard:
                frozen_time.tick(delta=datetime.timedelta(seconds=10.001))
        assert guard.remaining_timeout == pytest.approx(-0.001)

    def test_timeout_error_w_tuple_timeout(self, frozen_time):
        with pytest.raises(requests.exceptions.Timeout):
            with self.make_guard(timeout=(11, 10)) as guard:
                frozen_time.tick(delta=datetime.timedelta(seconds=10.001))
        assert guard.remaining_timeout == pytest.approx((0.999, -0.001))

    def test_custom_timeout_error_type(self, frozen_time):

        class FooError(Exception):
            pass
        with pytest.raises(FooError):
            with self.make_guard(timeout=1, timeout_error_type=FooError):
                frozen_time.tick(delta=datetime.timedelta(seconds=2))

    def test_lets_suite_errors_bubble_up(self, frozen_time):
        with pytest.raises(IndexError):
            with self.make_guard(timeout=1):
                [1, 2, 3][3]