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
def _test_relationships_in_order(self, obj_class):
    for field, versions in obj_class.obj_relationships.items():
        last_my_version = (0, 0)
        last_child_version = (0, 0)
        for my_version, child_version in versions:
            _my_version = vutils.convert_version_to_tuple(my_version)
            _ch_version = vutils.convert_version_to_tuple(child_version)
            if not (last_my_version < _my_version and last_child_version <= _ch_version):
                raise AssertionError('Object %s relationship %s->%s for field %s is out of order' % (obj_class.obj_name(), my_version, child_version, field))
            last_my_version = _my_version
            last_child_version = _ch_version