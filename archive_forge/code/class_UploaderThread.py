from __future__ import print_function
from boto.sdb.queryresultset import SelectResultSet
from boto.compat import six
import sys
from xml.sax.handler import ContentHandler
from threading import Thread
class UploaderThread(Thread):
    """Uploader Thread"""

    def __init__(self, domain):
        self.db = domain
        self.items = {}
        super(UploaderThread, self).__init__()

    def run(self):
        try:
            self.db.batch_put_attributes(self.items)
        except:
            print('Exception using batch put, trying regular put instead')
            for item_name in self.items:
                self.db.put_attributes(item_name, self.items[item_name])
        print('.', end=' ')
        sys.stdout.flush()