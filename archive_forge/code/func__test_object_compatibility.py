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
def _test_object_compatibility(self, obj_class, manifest=None, init_args=None, init_kwargs=None):
    init_args = init_args or []
    init_kwargs = init_kwargs or {}
    version = vutils.convert_version_to_tuple(obj_class.VERSION)
    kwargs = {'version_manifest': manifest} if manifest else {}
    for n in range(version[1] + 1):
        test_version = '%d.%d' % (version[0], n)
        LOG.debug('testing obj: %s version: %s' % (obj_class.obj_name(), test_version))
        kwargs['target_version'] = test_version
        obj_class(*init_args, **init_kwargs).obj_to_primitive(**kwargs)