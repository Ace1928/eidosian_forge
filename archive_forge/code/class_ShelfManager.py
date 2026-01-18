import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
class ShelfManager:
    """Maintain a list of shelved changes."""

    def __init__(self, tree, transport):
        self.tree = tree
        self.transport = transport.clone('shelf')
        self.transport.ensure_base()

    def get_shelf_filename(self, shelf_id):
        return 'shelf-%d' % shelf_id

    def get_shelf_ids(self, filenames):
        matcher = re.compile('shelf-([1-9][0-9]*)')
        shelf_ids = []
        for filename in filenames:
            match = matcher.match(filename)
            if match is not None:
                shelf_ids.append(int(match.group(1)))
        return shelf_ids

    def new_shelf(self):
        """Return a file object and id for a new set of shelved changes."""
        last_shelf = self.last_shelf()
        if last_shelf is None:
            next_shelf = 1
        else:
            next_shelf = last_shelf + 1
        filename = self.get_shelf_filename(next_shelf)
        shelf_file = open(self.transport.local_abspath(filename), 'wb')
        return (next_shelf, shelf_file)

    def shelve_changes(self, creator, message=None):
        """Store the changes in a ShelfCreator on a shelf."""
        next_shelf, shelf_file = self.new_shelf()
        try:
            creator.write_shelf(shelf_file, message)
        finally:
            shelf_file.close()
        creator.transform()
        return next_shelf

    def read_shelf(self, shelf_id):
        """Return the file associated with a shelf_id for reading.

        :param shelf_id: The id of the shelf to retrive the file for.
        """
        filename = self.get_shelf_filename(shelf_id)
        try:
            return open(self.transport.local_abspath(filename), 'rb')
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise
            raise NoSuchShelfId(shelf_id)

    def get_unshelver(self, shelf_id):
        """Return an unshelver for a given shelf_id.

        :param shelf_id: The shelf id to return the unshelver for.
        """
        shelf_file = self.read_shelf(shelf_id)
        try:
            return Unshelver.from_tree_and_shelf(self.tree, shelf_file)
        finally:
            shelf_file.close()

    def get_metadata(self, shelf_id):
        """Return the metadata associated with a given shelf_id."""
        shelf_file = self.read_shelf(shelf_id)
        try:
            records = Unshelver.iter_records(shelf_file)
        finally:
            shelf_file.close()
        return Unshelver.parse_metadata(records)

    def delete_shelf(self, shelf_id):
        """Delete the shelved changes for a given id.

        :param shelf_id: id of the shelved changes to delete.
        """
        filename = self.get_shelf_filename(shelf_id)
        self.transport.delete(filename)

    def active_shelves(self):
        """Return a list of shelved changes."""
        active = sorted(self.get_shelf_ids(self.transport.list_dir('.')))
        return active

    def last_shelf(self):
        """Return the id of the last-created shelved change."""
        active = self.active_shelves()
        if len(active) > 0:
            return active[-1]
        else:
            return None