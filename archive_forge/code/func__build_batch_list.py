import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
def _build_batch_list(self, params, items, replace=False):
    item_names = items.keys()
    i = 0
    for item_name in item_names:
        params['Item.%d.ItemName' % i] = item_name
        j = 0
        item = items[item_name]
        if item is not None:
            attr_names = item.keys()
            for attr_name in attr_names:
                value = item[attr_name]
                if isinstance(value, list):
                    for v in value:
                        if self.converter:
                            v = self.converter.encode(v)
                        params['Item.%d.Attribute.%d.Name' % (i, j)] = attr_name
                        params['Item.%d.Attribute.%d.Value' % (i, j)] = v
                        if replace:
                            params['Item.%d.Attribute.%d.Replace' % (i, j)] = 'true'
                        j += 1
                else:
                    params['Item.%d.Attribute.%d.Name' % (i, j)] = attr_name
                    if self.converter:
                        value = self.converter.encode(value)
                    params['Item.%d.Attribute.%d.Value' % (i, j)] = value
                    if replace:
                        params['Item.%d.Attribute.%d.Replace' % (i, j)] = 'true'
                    j += 1
        i += 1