import collections
import functools
import hashlib
import itertools
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from urllib import parse as urlparse
import yaql
from yaql.language import exceptions
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import function
def _result_ready(self, r):
    if r.action in (r.CREATE, r.ADOPT, r.SUSPEND, r.RESUME, r.UPDATE, r.ROLLBACK, r.SNAPSHOT, r.CHECK):
        return True
    return False