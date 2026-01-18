from collections import namedtuple
from collections import OrderedDict
import copy
import datetime
import inspect
import logging
from unittest import mock
import fixtures
from oslo_utils.secretutils import md5
from oslo_utils import versionutils as vutils
from oslo_versionedobjects import base
from oslo_versionedobjects import fields
class StableObjectJsonFixture(fixtures.Fixture):
    """Fixture that makes sure we get stable JSON object representations.

    Since objects contain things like set(), which can't be converted to
    JSON, we have some situations where the representation isn't fully
    deterministic. This doesn't matter at all at runtime, but does to
    unit tests that try to assert things at a low level.

    This fixture mocks the obj_to_primitive() call and makes sure to
    sort the list of changed fields (which came from a set) before
    returning it to the caller.
    """

    def __init__(self):
        self._original_otp = base.VersionedObject.obj_to_primitive

    def setUp(self):
        super(StableObjectJsonFixture, self).setUp()

        def _doit(obj, *args, **kwargs):
            result = self._original_otp(obj, *args, **kwargs)
            changes_key = obj._obj_primitive_key('changes')
            if changes_key in result:
                result[changes_key].sort()
            return result
        self.useFixture(fixtures.MonkeyPatch('oslo_versionedobjects.base.VersionedObject.obj_to_primitive', _doit))