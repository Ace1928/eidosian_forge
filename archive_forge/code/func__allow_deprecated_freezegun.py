from unittest import mock
import datetime
import duet
import pytest
import freezegun
import numpy as np
from google.protobuf.duration_pb2 import Duration
from google.protobuf.text_format import Merge
from google.protobuf.timestamp_pb2 import Timestamp
import cirq
import cirq_google as cg
from cirq_google.api import v2
from cirq_google.engine import util
from cirq_google.engine.engine import EngineContext
from cirq_google.cloud import quantum
def _allow_deprecated_freezegun(func):

    def wrapper(*args, **kwargs):
        import os
        from cirq.testing.deprecation import ALLOW_DEPRECATION_IN_TEST
        orig_exist, orig_value = (ALLOW_DEPRECATION_IN_TEST in os.environ, os.environ.get(ALLOW_DEPRECATION_IN_TEST, None))
        os.environ[ALLOW_DEPRECATION_IN_TEST] = 'True'
        try:
            return func(*args, **kwargs)
        finally:
            if orig_exist:
                os.environ[ALLOW_DEPRECATION_IN_TEST] = orig_value
            else:
                del os.environ[ALLOW_DEPRECATION_IN_TEST]
    return wrapper