import copy
import hmac
import uuid
import base64
import datetime
from hashlib import sha1
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import ET, b, httplib, urlencode
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.aws import AWSGenericResponse, AWSTokenConnection
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import LibcloudError
def _update_single_value_record(self, record, name=None, type=None, data=None, extra=None):
    batch = [('DELETE', record.name, record.type, record.data, record.extra), ('CREATE', name, type, data, extra)]
    return self._post_changeset(record.zone, batch)