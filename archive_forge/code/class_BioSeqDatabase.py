import os
from . import BioSeq
from . import Loader
from . import DBUtils
class BioSeqDatabase:
    """Represents a namespace (sub-database) within the BioSQL database.

    i.e. One row in the biodatabase table, and all all rows in the bioentry
    table associated with it.
    """

    def __init__(self, adaptor, name):
        """Create a BioDatabase object.

        Arguments:
         - adaptor - A BioSQL.Adaptor object
         - name - The name of the sub-database (namespace)

        """
        self.adaptor = adaptor
        self.name = name
        self.dbid = self.adaptor.fetch_dbid_by_dbname(name)

    def __repr__(self):
        """Return a short summary of the BioSeqDatabase."""
        return f'BioSeqDatabase({self.adaptor!r}, {self.name!r})'

    def get_Seq_by_id(self, name):
        """Get a DBSeqRecord object by its name.

        Example: seq_rec = db.get_Seq_by_id('ROA1_HUMAN')

        The name of this method is misleading since it returns a DBSeqRecord
        rather than a Seq object, and presumably was to mirror BioPerl.
        """
        seqid = self.adaptor.fetch_seqid_by_display_id(self.dbid, name)
        return BioSeq.DBSeqRecord(self.adaptor, seqid)

    def get_Seq_by_acc(self, name):
        """Get a DBSeqRecord object by accession number.

        Example: seq_rec = db.get_Seq_by_acc('X77802')

        The name of this method is misleading since it returns a DBSeqRecord
        rather than a Seq object, and presumably was to mirror BioPerl.
        """
        seqid = self.adaptor.fetch_seqid_by_accession(self.dbid, name)
        return BioSeq.DBSeqRecord(self.adaptor, seqid)

    def get_Seq_by_ver(self, name):
        """Get a DBSeqRecord object by version number.

        Example: seq_rec = db.get_Seq_by_ver('X77802.1')

        The name of this method is misleading since it returns a DBSeqRecord
        rather than a Seq object, and presumably was to mirror BioPerl.
        """
        seqid = self.adaptor.fetch_seqid_by_version(self.dbid, name)
        return BioSeq.DBSeqRecord(self.adaptor, seqid)

    def get_Seqs_by_acc(self, name):
        """Get a list of DBSeqRecord objects by accession number.

        Example: seq_recs = db.get_Seq_by_acc('X77802')

        The name of this method is misleading since it returns a list of
        DBSeqRecord objects rather than a list of Seq objects, and presumably
        was to mirror BioPerl.
        """
        seqids = self.adaptor.fetch_seqids_by_accession(self.dbid, name)
        return [BioSeq.DBSeqRecord(self.adaptor, seqid) for seqid in seqids]

    def __getitem__(self, key):
        """Return a DBSeqRecord for one of the sequences in the sub-database.

        Arguments:
         - key - The internal id for the sequence

        """
        record = BioSeq.DBSeqRecord(self.adaptor, key)
        if record._biodatabase_id != self.dbid:
            raise KeyError(f'Entry {key!r} does exist, but not in current name space')
        return record

    def __delitem__(self, key):
        """Remove an entry and all its annotation."""
        if key not in self:
            raise KeyError(f'Entry {key!r} cannot be deleted. It was not found or is invalid')
        sql = 'DELETE FROM bioentry WHERE biodatabase_id=%s AND bioentry_id=%s;'
        self.adaptor.execute(sql, (self.dbid, key))

    def __len__(self):
        """Return number of records in this namespace (sub database)."""
        sql = 'SELECT COUNT(bioentry_id) FROM bioentry WHERE biodatabase_id=%s;'
        return int(self.adaptor.execute_and_fetch_col0(sql, (self.dbid,))[0])

    def __contains__(self, value):
        """Check if a primary (internal) id is this namespace (sub database)."""
        sql = 'SELECT COUNT(bioentry_id) FROM bioentry WHERE biodatabase_id=%s AND bioentry_id=%s;'
        try:
            bioentry_id = int(value)
        except ValueError:
            return False
        return bool(self.adaptor.execute_and_fetch_col0(sql, (self.dbid, bioentry_id))[0])

    def __iter__(self):
        """Iterate over ids (which may not be meaningful outside this database)."""
        return iter(self.adaptor.list_bioentry_ids(self.dbid))

    def keys(self):
        """Iterate over ids (which may not be meaningful outside this database)."""
        return iter(self)

    def values(self):
        """Iterate over DBSeqRecord objects in the namespace (sub database)."""
        for key in self:
            yield self[key]

    def items(self):
        """Iterate over (id, DBSeqRecord) for the namespace (sub database)."""
        for key in self:
            yield (key, self[key])

    def lookup(self, **kwargs):
        """Return a DBSeqRecord using an acceptable identifier.

        Arguments:
         - kwargs - A single key-value pair where the key is one
           of primary_id, gi, display_id, name, accession, version

        """
        if len(kwargs) != 1:
            raise TypeError('single key/value parameter expected')
        k, v = list(kwargs.items())[0]
        if k not in _allowed_lookups:
            raise TypeError(f'lookup() expects one of {list(_allowed_lookups.keys())!r}, not {k!r}')
        lookup_name = _allowed_lookups[k]
        lookup_func = getattr(self.adaptor, lookup_name)
        seqid = lookup_func(self.dbid, v)
        return BioSeq.DBSeqRecord(self.adaptor, seqid)

    def load(self, record_iterator, fetch_NCBI_taxonomy=False):
        """Load a set of SeqRecords into the BioSQL database.

        record_iterator is either a list of SeqRecord objects, or an
        Iterator object that returns SeqRecord objects (such as the
        output from the Bio.SeqIO.parse() function), which will be
        used to populate the database.

        fetch_NCBI_taxonomy is boolean flag allowing or preventing
        connection to the taxonomic database on the NCBI server
        (via Bio.Entrez) to fetch a detailed taxonomy for each
        SeqRecord.

        Example::

            from Bio import SeqIO
            count = db.load(SeqIO.parse(open(filename), format))

        Returns the number of records loaded.
        """
        db_loader = Loader.DatabaseLoader(self.adaptor, self.dbid, fetch_NCBI_taxonomy)
        num_records = 0
        global _POSTGRES_RULES_PRESENT
        for cur_record in record_iterator:
            num_records += 1
            if _POSTGRES_RULES_PRESENT:
                if cur_record.id.count('.') == 1:
                    accession, version = cur_record.id.split('.')
                    try:
                        version = int(version)
                    except ValueError:
                        accession = cur_record.id
                        version = 0
                else:
                    accession = cur_record.id
                    version = 0
                gi = cur_record.annotations.get('gi')
                sql = "SELECT bioentry_id FROM bioentry WHERE (identifier = '%s' AND biodatabase_id = '%s') OR (accession = '%s' AND version = '%s' AND biodatabase_id = '%s')"
                self.adaptor.execute(sql % (gi, self.dbid, accession, version, self.dbid))
                if self.adaptor.cursor.fetchone():
                    raise self.adaptor.conn.IntegrityError('Duplicate record detected: record has not been inserted')
            db_loader.load_seqrecord(cur_record)
        return num_records