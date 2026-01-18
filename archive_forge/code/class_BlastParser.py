from Bio.Blast import Record
import xml.sax
from xml.sax.handler import ContentHandler
class BlastParser(_XMLparser):
    """Parse XML BLAST data into a Record.Blast object.

    Parses XML output from BLAST (direct use discouraged).
    This (now) returns a list of Blast records.
    Historically it returned a single Blast record.
    You are expected to use this via the parse or read functions.

    All XML 'action' methods are private methods and may be:

    - ``_start_TAG`` called when the start tag is found
    - ``_end_TAG`` called when the end tag is found

    """

    def __init__(self, debug=0):
        """Initialize the parser.

        Arguments:
         - debug - integer, amount of debug information to print

        """
        _XMLparser.__init__(self, debug)
        self._parser = xml.sax.make_parser()
        self._parser.setContentHandler(self)
        self._parser.setFeature(xml.sax.handler.feature_validation, 0)
        self._parser.setFeature(xml.sax.handler.feature_namespaces, 0)
        self._parser.setFeature(xml.sax.handler.feature_external_pes, 0)
        self._parser.setFeature(xml.sax.handler.feature_external_ges, 0)
        self._xml_version = 1
        self.reset()

    def reset(self):
        """Reset all the data allowing reuse of the BlastParser() object."""
        self._records = []
        self._header = Record.Header()
        self._parameters = Record.Parameters()
        self._parameters.filter = None

    def _on_root_node(self, name):
        if name == 'BlastOutput':
            self._setup_blast_v1()
        elif name == 'BlastXML2':
            self._setup_blast_v2()
        else:
            raise ValueError('Invalid root node name: %s. Root node should be either BlastOutput or BlastXML2' % name)

    def _setup_blast_v1(self):
        self._method_map = {'start_Iteration': self._start_blast_record, 'end_Iteration': self._end_blast_record, 'end_BlastOutput_program': self._set_header_application, 'end_BlastOutput_version': self._set_header_version, 'end_BlastOutput_reference': self._set_header_reference, 'end_BlastOutput_db': self._set_header_database, 'end_BlastOutput_query-ID': self._set_header_query_id, 'end_BlastOutput_query-def': self._set_header_query, 'end_BlastOutput_query-len': self._set_header_query_letters, 'end_Iteration_query-ID': self._set_record_query_id, 'end_Iteration_query-def': self._set_record_query_def, 'end_Iteration_query-len': self._set_record_query_letters, 'end_BlastOutput_hits': self._set_record_hits, 'end_Parameters_matrix': self._set_parameters_matrix, 'end_Parameters_expect': self._set_parameters_expect, 'end_Parameters_sc-match': self._set_parameters_sc_match, 'end_Parameters_sc-mismatch': self._set_parameters_sc_mismatch, 'end_Parameters_gap-open': self._set_parameters_gap_penalties, 'end_Parameters_gap-extend': self._set_parameters_gap_extend, 'end_Parameters_filter': self._set_parameters_filter, 'start_Hit': self._start_hit, 'end_Hit': self._end_hit, 'end_Hit_id': self.set_hit_id, 'end_Hit_def': self.set_hit_def, 'end_Hit_accession': self.set_hit_accession, 'end_Hit_len': self.set_hit_len, 'start_Hsp': self._start_hsp, 'end_Hsp_score': self._set_hsp_score, 'end_Hsp_bit-score': self._set_hsp_bit_score, 'end_Hsp_evalue': self._set_hsp_e_value, 'end_Hsp_query-from': self._set_hsp_query_start, 'end_Hsp_query-to': self._set_hsp_query_end, 'end_Hsp_hit-from': self._set_hsp_hit_from, 'end_Hsp_hit-to': self._set_hsp_hit_to, 'end_Hsp_query-frame': self._set_hsp_query_frame, 'end_Hsp_hit-frame': self._set_hsp_hit_frame, 'end_Hsp_identity': self._set_hsp_identity, 'end_Hsp_positive': self._set_hsp_positive, 'end_Hsp_gaps': self._set_hsp_gaps, 'end_Hsp_align-len': self._set_hsp_align_len, 'end_Hsp_qseq': self._set_hsp_query_seq, 'end_Hsp_hseq': self._set_hsp_subject_seq, 'end_Hsp_midline': self._set_hsp_midline, 'end_Statistics_db-num': self._set_statistics_db_num, 'end_Statistics_db-len': self._set_statistics_db_len, 'end_Statistics_hsp-len': self._set_statistics_hsp_len, 'end_Statistics_eff-space': self._set_statistics_eff_space, 'end_Statistics_kappa': self._set_statistics_kappa, 'end_Statistics_lambda': self._set_statistics_lambda, 'end_Statistics_entropy': self._set_statistics_entropy}

    def _setup_blast_v2(self):
        self._method_name_level = 2
        self._xml_version = 2
        self._method_map = {'start_report/Report': self._start_blast_record, 'end_report/Report': self._end_blast_record, 'end_Report/program': self._set_header_application, 'end_Report/version': self._set_header_version, 'end_Report/reference': self._set_header_reference, 'end_Target/db': self._set_header_database, 'end_Search/query-id': self._set_record_query_id, 'end_Search/query-title': self._set_record_query_def, 'end_Search/query-len': self._set_record_query_letters, 'end_BlastOutput_hits': self._set_record_hits, 'end_Parameters/matrix': self._set_parameters_matrix, 'end_Parameters/expect': self._set_parameters_expect, 'end_Parameters/sc-match': self._set_parameters_sc_match, 'end_Parameters/sc-mismatch': self._set_parameters_sc_mismatch, 'end_Parameters/gap-open': self._set_parameters_gap_penalties, 'end_Parameters/gap-extend': self._set_parameters_gap_extend, 'end_Parameters/filter': self._set_parameters_filter, 'start_hits/Hit': self._start_hit, 'end_hits/Hit': self._end_hit, 'start_description/HitDescr': self._start_hit_descr_item, 'end_description/HitDescr': self._end_hit_descr_item, 'end_HitDescr/id': self._end_description_id, 'end_HitDescr/accession': self._end_description_accession, 'end_HitDescr/title': self._end_description_title, 'end_HitDescr/taxid': self._end_description_taxid, 'end_HitDescr/sciname': self._end_description_sciname, 'end_Hit/len': self.set_hit_len, 'start_hsps/Hsp': self._start_hsp, 'end_hsps/Hsp': self._end_hsp, 'end_Hsp/score': self._set_hsp_score, 'end_Hsp/bit-score': self._set_hsp_bit_score, 'end_Hsp/evalue': self._set_hsp_e_value, 'end_Hsp/query-from': self._set_hsp_query_start, 'end_Hsp/query-to': self._set_hsp_query_end, 'end_Hsp/hit-from': self._set_hsp_hit_from, 'end_Hsp/hit-to': self._set_hsp_hit_to, 'end_Hsp/query-frame': self._set_hsp_query_frame, 'end_Hsp/hit-frame': self._set_hsp_hit_frame, 'end_Hsp/query-strand': self._set_hsp_query_strand, 'end_Hsp/hit-strand': self._set_hsp_hit_strand, 'end_Hsp/identity': self._set_hsp_identity, 'end_Hsp/positive': self._set_hsp_positive, 'end_Hsp/gaps': self._set_hsp_gaps, 'end_Hsp/align-len': self._set_hsp_align_len, 'end_Hsp/qseq': self._set_hsp_query_seq, 'end_Hsp/hseq': self._set_hsp_subject_seq, 'end_Hsp/midline': self._set_hsp_midline, 'end_Statistics/db-num': self._set_statistics_db_num, 'end_Statistics/db-len': self._set_statistics_db_len, 'end_Statistics/hsp-len': self._set_statistics_hsp_len, 'end_Statistics/eff-space': self._set_statistics_eff_space, 'end_Statistics/kappa': self._set_statistics_kappa, 'end_Statistics/lambda': self._set_statistics_lambda, 'end_Statistics/entropy': self._set_statistics_entropy}

    def _start_blast_record(self):
        """Start interaction (PRIVATE)."""
        self._blast = Record.Blast()

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

    def _set_header_application(self):
        """BLAST program, e.g., blastp, blastn, etc. (PRIVATE).

        Save this to put on each blast record object
        """
        self._header.application = self._value.upper()

    def _set_header_version(self):
        """Version number and date of the BLAST engine (PRIVATE).

        e.g. "BLASTX 2.2.12 [Aug-07-2005]" but there can also be
        variants like "BLASTP 2.2.18+" without the date.

        Save this to put on each blast record object
        """
        parts = self._value.split()
        self._header.version = parts[1]
        if len(parts) >= 3:
            if parts[2][0] == '[' and parts[2][-1] == ']':
                self._header.date = parts[2][1:-1]
            else:
                self._header.date = parts[2]

    def _set_header_reference(self):
        """Record any article reference describing the algorithm (PRIVATE).

        Save this to put on each blast record object
        """
        self._header.reference = self._value

    def _set_header_database(self):
        """Record the database(s) searched (PRIVATE).

        Save this to put on each blast record object
        """
        self._header.database = self._value

    def _set_header_query_id(self):
        """Record the identifier of the query (PRIVATE).

        Important in old pre 2.2.14 BLAST, for recent versions
        <Iteration_query-ID> is enough
        """
        self._header.query_id = self._value

    def _set_header_query(self):
        """Record the definition line of the query (PRIVATE).

        Important in old pre 2.2.14 BLAST, for recent versions
        <Iteration_query-def> is enough
        """
        self._header.query = self._value

    def _set_header_query_letters(self):
        """Record the length of the query (PRIVATE).

        Important in old pre 2.2.14 BLAST, for recent versions
        <Iteration_query-len> is enough
        """
        self._header.query_letters = int(self._value)

    def _set_record_query_id(self):
        """Record the identifier of the query (PRIVATE)."""
        self._blast.query_id = self._value

    def _set_record_query_def(self):
        """Record the definition line of the query (PRIVATE)."""
        self._blast.query = self._value

    def _set_record_query_letters(self):
        """Record the length of the query (PRIVATE)."""
        self._blast.query_letters = int(self._value)

    def _set_record_hits(self):
        """Hits to the database sequences, one for every sequence (PRIVATE)."""
        self._blast.num_hits = int(self._value)

    def _set_parameters_matrix(self):
        """Matrix used (-M on legacy BLAST) (PRIVATE)."""
        self._parameters.matrix = self._value

    def _set_parameters_expect(self):
        """Expect values cutoff (PRIVATE)."""
        self._parameters.expect = self._value

    def _set_parameters_sc_match(self):
        """Match score for nucleotide-nucleotide comparison (-r) (PRIVATE)."""
        self._parameters.sc_match = int(self._value)

    def _set_parameters_sc_mismatch(self):
        """Mismatch penalty for nucleotide-nucleotide comparison (-r) (PRIVATE)."""
        self._parameters.sc_mismatch = int(self._value)

    def _set_parameters_gap_penalties(self):
        """Gap existence cost (-G) (PRIVATE)."""
        self._parameters.gap_penalties = int(self._value)

    def _set_parameters_gap_extend(self):
        """Gap extension cose (-E) (PRIVATE)."""
        self._parameters.gap_penalties = (self._parameters.gap_penalties, int(self._value))

    def _set_parameters_filter(self):
        """Record filtering options (-F) (PRIVATE)."""
        self._parameters.filter = self._value

    def _start_hit(self):
        """Start filling records (PRIVATE)."""
        self._blast.alignments.append(Record.Alignment())
        self._descr = Record.Description() if self._xml_version == 1 else Record.DescriptionExt()
        self._blast.descriptions.append(self._descr)
        self._blast.multiple_alignment = []
        self._hit = self._blast.alignments[-1]
        self._descr.num_alignments = 0
        if self._value.strip() == 'CREATE_VIEW':
            print(f'NCBIXML: Ignored: {self._value!r}')
            self._value = ''

    def _end_hit(self):
        """Clear variables (PRIVATE)."""
        self._blast.multiple_alignment = None
        self._hit = None
        self._descr = None

    def set_hit_id(self):
        """Record the identifier of the database sequence (PRIVATE)."""
        self._hit.hit_id = self._value
        self._hit.title = self._value + ' '

    def set_hit_def(self):
        """Record the definition line of the database sequence (PRIVATE)."""
        self._hit.hit_def = self._value
        self._hit.title += self._value
        self._descr.title = self._hit.title

    def set_hit_accession(self):
        """Record the accession value of the database sequence (PRIVATE)."""
        self._hit.accession = self._value
        self._descr.accession = self._value

    def set_hit_len(self):
        """Record the length of the hit."""
        self._hit.length = int(self._value)

    def _start_hsp(self):
        self._hsp = Record.HSP()
        self._hsp.positives = None
        self._hit.hsps.append(self._hsp)
        self._descr.num_alignments += 1
        self._blast.multiple_alignment.append(Record.MultipleAlignment())
        self._mult_al = self._blast.multiple_alignment[-1]

    def _end_hsp(self):
        if self._hsp.frame and len(self._hsp.frame) == 1:
            self._hsp.frame += (0,)

    def _set_hsp_score(self):
        """Record the raw score of HSP (PRIVATE)."""
        self._hsp.score = float(self._value)
        if self._descr.score is None:
            self._descr.score = float(self._value)

    def _set_hsp_bit_score(self):
        """Record the Bit score of HSP (PRIVATE)."""
        self._hsp.bits = float(self._value)
        if self._descr.bits is None:
            self._descr.bits = float(self._value)

    def _set_hsp_e_value(self):
        """Record the expect value of the HSP (PRIVATE)."""
        self._hsp.expect = float(self._value)
        if self._descr.e is None:
            self._descr.e = float(self._value)

    def _set_hsp_query_start(self):
        """Offset of query at the start of the alignment (one-offset) (PRIVATE)."""
        self._hsp.query_start = int(self._value)

    def _set_hsp_query_end(self):
        """Offset of query at the end of the alignment (one-offset) (PRIVATE)."""
        self._hsp.query_end = int(self._value)

    def _set_hsp_hit_from(self):
        """Offset of the database at the start of the alignment (one-offset) (PRIVATE)."""
        self._hsp.sbjct_start = int(self._value)

    def _set_hsp_hit_to(self):
        """Offset of the database at the end of the alignment (one-offset) (PRIVATE)."""
        self._hsp.sbjct_end = int(self._value)

    def _set_hsp_query_frame(self):
        """Frame of the query if applicable (PRIVATE)."""
        v = int(self._value)
        self._hsp.frame = (v,)
        if self._header.application == 'BLASTN':
            self._hsp.strand = ('Plus' if v > 0 else 'Minus',)

    def _set_hsp_hit_frame(self):
        """Frame of the database sequence if applicable (PRIVATE)."""
        v = int(self._value)
        if len(self._hsp.frame) == 0:
            self._hsp.frame = (0, v)
        else:
            self._hsp.frame += (v,)
        if self._header.application == 'BLASTN':
            self._hsp.strand += ('Plus' if v > 0 else 'Minus',)

    def _set_hsp_query_strand(self):
        """Frame of the query if applicable (PRIVATE)."""
        self._hsp.strand = (self._value,)
        if self._header.application == 'BLASTN':
            self._hsp.frame = (1 if self._value == 'Plus' else -1,)

    def _set_hsp_hit_strand(self):
        """Frame of the database sequence if applicable (PRIVATE)."""
        self._hsp.strand += (self._value,)
        if self._header.application == 'BLASTN':
            self._hsp.frame += (1 if self._value == 'Plus' else -1,)

    def _set_hsp_identity(self):
        """Record the number of identities in the alignment (PRIVATE)."""
        v = int(self._value)
        self._hsp.identities = v
        if self._hsp.positives is None:
            self._hsp.positives = v

    def _set_hsp_positive(self):
        """Record the number of positive (conservative) substitutions in the alignment (PRIVATE)."""
        self._hsp.positives = int(self._value)

    def _set_hsp_gaps(self):
        """Record the number of gaps in the alignment (PRIVATE)."""
        self._hsp.gaps = int(self._value)

    def _set_hsp_align_len(self):
        """Record the length of the alignment (PRIVATE)."""
        self._hsp.align_length = int(self._value)

    def _set_hsp_query_seq(self):
        """Record the alignment string for the query (PRIVATE)."""
        self._hsp.query = self._value

    def _set_hsp_subject_seq(self):
        """Record the alignment string for the database (PRIVATE)."""
        self._hsp.sbjct = self._value

    def _set_hsp_midline(self):
        """Record the middle line as normally seen in BLAST report (PRIVATE)."""
        self._hsp.match = self._value
        assert len(self._hsp.match) == len(self._hsp.query)
        assert len(self._hsp.match) == len(self._hsp.sbjct)

    def _set_statistics_db_num(self):
        """Record the number of sequences in the database (PRIVATE)."""
        self._blast.num_sequences_in_database = int(self._value)

    def _set_statistics_db_len(self):
        """Record the number of letters in the database (PRIVATE)."""
        self._blast.num_letters_in_database = int(self._value)

    def _set_statistics_hsp_len(self):
        """Record the effective HSP length (PRIVATE)."""
        self._blast.effective_hsp_length = int(self._value)

    def _set_statistics_eff_space(self):
        """Record the effective search space (PRIVATE)."""
        self._blast.effective_search_space = float(self._value)

    def _set_statistics_kappa(self):
        """Karlin-Altschul parameter K (PRIVATE)."""
        self._blast.ka_params = float(self._value)

    def _set_statistics_lambda(self):
        """Karlin-Altschul parameter Lambda (PRIVATE)."""
        self._blast.ka_params = (float(self._value), self._blast.ka_params)

    def _set_statistics_entropy(self):
        """Karlin-Altschul parameter H (PRIVATE)."""
        self._blast.ka_params = self._blast.ka_params + (float(self._value),)

    def _start_hit_descr_item(self):
        """XML v2. Start hit description item."""
        self._hit_descr_item = Record.DescriptionExtItem()

    def _end_hit_descr_item(self):
        """XML v2. Start hit description item."""
        self._descr.append_item(self._hit_descr_item)
        if not self._hit.title:
            self._hit.title = str(self._hit_descr_item)
        self._hit_descr_item = None

    def _end_description_id(self):
        """XML v2. The identifier of the database sequence(PRIVATE)."""
        self._hit_descr_item.id = self._value
        if not self._hit.hit_id:
            self._hit.hit_id = self._value

    def _end_description_accession(self):
        """XML v2. The accession value of the database sequence (PRIVATE)."""
        self._hit_descr_item.accession = self._value
        if not getattr(self._hit, 'accession', None):
            self._hit.accession = self._value

    def _end_description_title(self):
        """XML v2. The hit description title (PRIVATE)."""
        self._hit_descr_item.title = self._value

    def _end_description_taxid(self):
        try:
            self._hit_descr_item.taxid = int(self._value)
        except ValueError:
            pass

    def _end_description_sciname(self):
        self._hit_descr_item.sciname = self._value