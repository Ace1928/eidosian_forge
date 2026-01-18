import xml.sax
import threading
import boto
from boto import handler
from boto.connection import AWSQueryConnection
from boto.sdb.domain import Domain, DomainMetaData
from boto.sdb.item import Item
from boto.sdb.regioninfo import SDBRegionInfo
from boto.exception import SDBResponseError
class ItemThread(threading.Thread):
    """
    A threaded :class:`Item <boto.sdb.item.Item>` retriever utility class.
    Retrieved :class:`Item <boto.sdb.item.Item>` objects are stored in the
    ``items`` instance variable after :py:meth:`run() <run>` is called.

    .. tip:: The item retrieval will not start until
        the :func:`run() <boto.sdb.connection.ItemThread.run>` method is called.
    """

    def __init__(self, name, domain_name, item_names):
        """
        :param str name: A thread name. Used for identification.
        :param str domain_name: The name of a SimpleDB
            :class:`Domain <boto.sdb.domain.Domain>`
        :type item_names: string or list of strings
        :param item_names: The name(s) of the items to retrieve from the specified
            :class:`Domain <boto.sdb.domain.Domain>`.
        :ivar list items: A list of items retrieved. Starts as empty list.
        """
        super(ItemThread, self).__init__(name=name)
        self.domain_name = domain_name
        self.conn = SDBConnection()
        self.item_names = item_names
        self.items = []

    def run(self):
        """
        Start the threaded retrieval of items. Populates the
        ``items`` list with :class:`Item <boto.sdb.item.Item>` objects.
        """
        for item_name in self.item_names:
            item = self.conn.get_attributes(self.domain_name, item_name)
            self.items.append(item)