import os.path as _path
import csv as _csv
from netaddr.compat import _open_binary
from netaddr.core import Subscriber, Publisher
class FileIndexer(Subscriber):
    """
    A concrete Subscriber that receives OUI record offset information that is
    written to an index data file as a set of comma separated records.
    """

    def __init__(self, index_file):
        """
        Constructor.

        :param index_file: a file-like object or name of index file where
            index records will be written.
        """
        if hasattr(index_file, 'readline') and hasattr(index_file, 'tell'):
            self.fh = index_file
        else:
            self.fh = open(index_file, 'w')
        self.writer = _csv.writer(self.fh, lineterminator='\n')

    def update(self, data):
        """
        Receives and writes index data to a CSV data file.

        :param data: record containing offset record information.
        """
        self.writer.writerow(data)