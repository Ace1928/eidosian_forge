import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
@base.remotable
def _update_test(self):
    project_id = getattr(context, 'tenant', None)
    if project_id is None:
        project_id = getattr(context, 'project_id', None)
    if project_id == 'alternate':
        self.bar = 'alternate-context'
    else:
        self.bar = 'updated'