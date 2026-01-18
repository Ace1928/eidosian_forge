import abc
import datetime
from dogpile.cache import api
from dogpile import util as dp_util
from oslo_cache import core
from oslo_log import log
from oslo_utils import importutils
from oslo_utils import timeutils
from oslo_cache._i18n import _
from oslo_cache import exception
def _get_doc_date(self):
    if self.ttl_seconds > 0:
        expire_delta = datetime.timedelta(seconds=self.ttl_seconds)
        doc_date = timeutils.utcnow() + expire_delta
    else:
        doc_date = timeutils.utcnow()
    return doc_date