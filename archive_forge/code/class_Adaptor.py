import os
from . import BioSeq
from . import Loader
from . import DBUtils
class Adaptor:
    """High level wrapper for a database connection and cursor.

    Most database calls in BioSQL are done indirectly though this adaptor
    class. This provides helper methods for fetching data and executing
    sql.
    """

    def __init__(self, conn, dbutils, wrap_cursor=False):
        """Create an Adaptor object.

        Arguments:
         - conn - A database connection
         - dbutils - A BioSQL.DBUtils object
         - wrap_cursor - Optional, whether to wrap the cursor object

        """
        self.conn = conn
        if wrap_cursor:
            self.cursor = _CursorWrapper(conn.cursor())
        else:
            self.cursor = conn.cursor()
        self.dbutils = dbutils

    def last_id(self, table):
        """Return the last row id for the selected table."""
        return self.dbutils.last_id(self.cursor, table)

    def autocommit(self, y=True):
        """Set the autocommit mode. True values enable; False value disable."""
        return self.dbutils.autocommit(self.conn, y)

    def commit(self):
        """Commit the current transaction."""
        return self.conn.commit()

    def rollback(self):
        """Roll-back the current transaction."""
        return self.conn.rollback()

    def close(self):
        """Close the connection. No further activity possible."""
        return self.conn.close()

    def fetch_dbid_by_dbname(self, dbname):
        """Return the internal id for the sub-database using its name."""
        self.execute('select biodatabase_id from biodatabase where name = %s', (dbname,))
        rv = self.cursor.fetchall()
        if not rv:
            raise KeyError(f'Cannot find biodatabase with name {dbname!r}')
        return rv[0][0]

    def fetch_seqid_by_display_id(self, dbid, name):
        """Return the internal id for a sequence using its display id.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the name of the sequence. Corresponds to the
           name column of the bioentry table of the SQL schema

        """
        sql = 'select bioentry_id from bioentry where name = %s'
        fields = [name]
        if dbid:
            sql += ' and biodatabase_id = %s'
            fields.append(dbid)
        self.execute(sql, fields)
        rv = self.cursor.fetchall()
        if not rv:
            raise IndexError(f'Cannot find display id {name!r}')
        if len(rv) > 1:
            raise IndexError(f'More than one entry with display id {name!r}')
        return rv[0][0]

    def fetch_seqid_by_accession(self, dbid, name):
        """Return the internal id for a sequence using its accession.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the accession of the sequence. Corresponds to the
           accession column of the bioentry table of the SQL schema

        """
        sql = 'select bioentry_id from bioentry where accession = %s'
        fields = [name]
        if dbid:
            sql += ' and biodatabase_id = %s'
            fields.append(dbid)
        self.execute(sql, fields)
        rv = self.cursor.fetchall()
        if not rv:
            raise IndexError(f'Cannot find accession {name!r}')
        if len(rv) > 1:
            raise IndexError(f'More than one entry with accession {name!r}')
        return rv[0][0]

    def fetch_seqids_by_accession(self, dbid, name):
        """Return a list internal ids using an accession.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the accession of the sequence. Corresponds to the
           accession column of the bioentry table of the SQL schema

        """
        sql = 'select bioentry_id from bioentry where accession = %s'
        fields = [name]
        if dbid:
            sql += ' and biodatabase_id = %s'
            fields.append(dbid)
        return self.execute_and_fetch_col0(sql, fields)

    def fetch_seqid_by_version(self, dbid, name):
        """Return the internal id for a sequence using its accession and version.

        Arguments:
         - dbid - the internal id for the sub-database
         - name - the accession of the sequence containing a version number.
           Must correspond to <accession>.<version>

        """
        acc_version = name.split('.')
        if len(acc_version) > 2:
            raise IndexError(f'Bad version {name!r}')
        acc = acc_version[0]
        if len(acc_version) == 2:
            version = acc_version[1]
        else:
            version = '0'
        sql = 'SELECT bioentry_id FROM bioentry WHERE accession = %s AND version = %s'
        fields = [acc, version]
        if dbid:
            sql += ' and biodatabase_id = %s'
            fields.append(dbid)
        self.execute(sql, fields)
        rv = self.cursor.fetchall()
        if not rv:
            raise IndexError(f'Cannot find version {name!r}')
        if len(rv) > 1:
            raise IndexError(f'More than one entry with version {name!r}')
        return rv[0][0]

    def fetch_seqid_by_identifier(self, dbid, identifier):
        """Return the internal id for a sequence using its identifier.

        Arguments:
         - dbid - the internal id for the sub-database
         - identifier - the identifier of the sequence. Corresponds to
           the identifier column of the bioentry table in the SQL schema.

        """
        sql = 'SELECT bioentry_id FROM bioentry WHERE identifier = %s'
        fields = [identifier]
        if dbid:
            sql += ' and biodatabase_id = %s'
            fields.append(dbid)
        self.execute(sql, fields)
        rv = self.cursor.fetchall()
        if not rv:
            raise IndexError(f'Cannot find display id {identifier!r}')
        return rv[0][0]

    def list_biodatabase_names(self):
        """Return a list of all of the sub-databases."""
        return self.execute_and_fetch_col0('SELECT name FROM biodatabase')

    def list_bioentry_ids(self, dbid):
        """Return a list of internal ids for all of the sequences in a sub-databae.

        Arguments:
         - dbid - The internal id for a sub-database

        """
        return self.execute_and_fetch_col0('SELECT bioentry_id FROM bioentry WHERE biodatabase_id = %s', (dbid,))

    def list_bioentry_display_ids(self, dbid):
        """Return a list of all sequence names in a sub-databae.

        Arguments:
         - dbid - The internal id for a sub-database

        """
        return self.execute_and_fetch_col0('SELECT name FROM bioentry WHERE biodatabase_id = %s', (dbid,))

    def list_any_ids(self, sql, args):
        """Return ids given a SQL statement to select for them.

        This assumes that the given SQL does a SELECT statement that
        returns a list of items. This parses them out of the 2D list
        they come as and just returns them in a list.
        """
        return self.execute_and_fetch_col0(sql, args)

    def execute_one(self, sql, args=None):
        """Execute sql that returns 1 record, and return the record."""
        self.execute(sql, args or ())
        rv = self.cursor.fetchall()
        if len(rv) != 1:
            raise ValueError(f'Expected 1 response, got {len(rv)}.')
        return rv[0]

    def execute(self, sql, args=None):
        """Just execute an sql command."""
        if os.name == 'java':
            sql = sql.replace('%s', '?')
        self.dbutils.execute(self.cursor, sql, args)

    def executemany(self, sql, args):
        """Execute many sql commands."""
        if os.name == 'java':
            sql = sql.replace('%s', '?')
        self.dbutils.executemany(self.cursor, sql, args)

    def get_subseq_as_string(self, seqid, start, end):
        """Return a substring of a sequence.

        Arguments:
         - seqid - The internal id for the sequence
         - start - The start position of the sequence; 0-indexed
         - end - The end position of the sequence

        """
        length = end - start
        return self.execute_one('SELECT SUBSTR(seq, %s, %s) FROM biosequence WHERE bioentry_id = %s', (start + 1, length, seqid))[0]

    def execute_and_fetch_col0(self, sql, args=None):
        """Return a list of values from the first column in the row."""
        self.execute(sql, args or ())
        return [field[0] for field in self.cursor.fetchall()]

    def execute_and_fetchall(self, sql, args=None):
        """Return a list of tuples of all rows."""
        self.execute(sql, args or ())
        return self.cursor.fetchall()