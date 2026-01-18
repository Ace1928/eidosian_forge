from time import gmtime, strftime
from Bio.SeqUtils.CheckSum import crc64
from Bio import Entrez
from Bio.Seq import UndefinedSequenceError
from Bio.SeqFeature import UnknownPosition
def _insert_location(self, location, rank, seqfeature_id):
    """Add SeqFeature location to seqfeature_location table (PRIVATE).

        TODO - Add location operator to location_qualifier_value?
        """
    try:
        start = int(location.start) + 1
    except TypeError:
        if isinstance(location.start, UnknownPosition):
            start = None
        else:
            raise
    try:
        end = int(location.end)
    except TypeError:
        if isinstance(location.end, UnknownPosition):
            end = None
        else:
            raise
    strand = location.strand or 0
    loc_term_id = None
    if location.ref:
        dbxref_id = self._get_dbxref_id(location.ref_db or '', location.ref)
    else:
        dbxref_id = None
    sql = 'INSERT INTO location (seqfeature_id, dbxref_id, term_id,start_pos, end_pos, strand, "rank") VALUES (%s, %s, %s, %s, %s, %s, %s)'
    self.adaptor.execute(sql, (seqfeature_id, dbxref_id, loc_term_id, start, end, strand, rank))
    '\n        # See Bug 2677\n        # TODO - Record the location_operator (e.g. "join" or "order")\n        # using the location_qualifier_value table (which we and BioPerl\n        # have historically left empty).\n        # Note this will need an ontology term for the location qualifier\n        # (location_qualifier_value.term_id) for which oddly the schema\n        # does not allow NULL.\n        if feature.location_operator:\n            #e.g. "join" (common),\n            #or "order" (see Tests/GenBank/protein_refseq2.gb)\n            location_id = self.adaptor.last_id(\'location\')\n            loc_qual_term_id = None # Not allowed in BioSQL v1.0.1\n            sql = ("INSERT INTO location_qualifier_value"\n                   "(location_id, term_id, value) "\n                   "VALUES (%s, %s, %s)")\n            self.adaptor.execute(sql, (location_id, loc_qual_term_id,\n                                       feature.location_operator))\n        '