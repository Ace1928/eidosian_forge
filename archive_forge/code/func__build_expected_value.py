import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def _build_expected_value(self, params, expected_value):
    params['Expected.1.Name'] = expected_value[0]
    if expected_value[1] is True:
        params['Expected.1.Exists'] = 'true'
    elif expected_value[1] is False:
        params['Expected.1.Exists'] = 'false'
    else:
        params['Expected.1.Value'] = expected_value[1]