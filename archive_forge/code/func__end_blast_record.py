from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
def _end_blast_record(self):
    """End interaction (PRIVATE)."""
    self._blast.reference = self._header.reference
    self._blast.date = self._header.date
    self._blast.version = self._header.version
    self._blast.database = self._header.database
    self._blast.application = self._header.application
    if not hasattr(self._blast, 'query') or not self._blast.query:
        self._blast.query = self._header.query
    if not hasattr(self._blast, 'query_id') or not self._blast.query_id:
        self._blast.query_id = self._header.query_id
    if not hasattr(self._blast, 'query_letters') or not self._blast.query_letters:
        self._blast.query_letters = self._header.query_letters
    self._blast.query_length = self._blast.query_letters
    self._blast.database_length = self._blast.num_letters_in_database
    self._blast.database_sequences = self._blast.num_sequences_in_database
    self._blast.matrix = self._parameters.matrix
    self._blast.num_seqs_better_e = self._parameters.num_seqs_better_e
    self._blast.gap_penalties = self._parameters.gap_penalties
    self._blast.filter = self._parameters.filter
    self._blast.expect = self._parameters.expect
    self._blast.sc_match = self._parameters.sc_match
    self._blast.sc_mismatch = self._parameters.sc_mismatch
    self._records.append(self._blast)
    self._blast = None
    if self._debug:
        print('NCBIXML: Added Blast record to results')