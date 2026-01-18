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
class Route53DNSResponse(AWSGenericResponse):
    """
    Amazon Route53 response class.
    """
    namespace = NAMESPACE
    xpath = 'Error'
    exceptions = {'NoSuchHostedZone': ZoneDoesNotExistError, 'InvalidChangeBatch': InvalidChangeBatch}