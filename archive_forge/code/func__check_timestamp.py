import datetime
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from werkzeug import exceptions
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception as ks_exceptions
from keystone.i18n import _
from keystone.server import flask as ks_flask
@staticmethod
def _check_timestamp(credentials):
    timestamp = credentials.get('params', {}).get('Timestamp') or credentials.get('headers', {}).get('X-Amz-Date') or credentials.get('params', {}).get('X-Amz-Date')
    if not timestamp:
        return
    try:
        timestamp = timeutils.parse_isotime(timestamp)
        timestamp = timeutils.normalize_time(timestamp)
    except Exception as e:
        raise ks_exceptions.Unauthorized(_('Credential timestamp is invalid: %s') % e)
    auth_ttl = datetime.timedelta(minutes=CONF.credential.auth_ttl)
    current_time = timeutils.normalize_time(timeutils.utcnow())
    if current_time > timestamp + auth_ttl:
        raise ks_exceptions.Unauthorized(_('Credential is expired'))