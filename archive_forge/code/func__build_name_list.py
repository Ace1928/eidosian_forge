import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def _build_name_list(self, params, attribute_names):
    i = 1
    attribute_names.sort()
    for name in attribute_names:
        params['Attribute.%d.Name' % i] = name
        i += 1