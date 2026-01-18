import os.path
import shutil
import fixtures
import oslo_middleware
from glance.api.middleware import context
from glance.common import config
from glance.tests import utils as test_utils
def _appendto(orig, copy, str):
    shutil.copy(orig, copy)
    with open(copy, 'a') as f:
        f.write(str or '')
        f.flush()