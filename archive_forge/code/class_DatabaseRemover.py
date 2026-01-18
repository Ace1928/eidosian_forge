from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
class DatabaseRemover:
    """Complement the Loader functionality by fully removing a database.

    This probably isn't really useful for normal purposes, since you
    can just do a::

        DROP DATABASE db_name

    and then recreate the database. But, it's really useful for testing
    purposes.
    """

    def __init__(self, adaptor, dbid):
        """Initialize with a database id and adaptor connection."""
        self.adaptor = adaptor
        self.dbid = dbid

    def remove(self):
        """Remove everything related to the given database id."""
        sql = 'DELETE FROM bioentry WHERE biodatabase_id = %s'
        self.adaptor.execute(sql, (self.dbid,))
        sql = 'DELETE FROM biodatabase WHERE biodatabase_id = %s'
        self.adaptor.execute(sql, (self.dbid,))