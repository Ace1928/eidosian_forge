import re
import warnings
from Bio import BiopythonParserWarning
from Bio.Seq import Seq
from Bio.SeqFeature import Location
from Bio.SeqFeature import Reference
from Bio.SeqFeature import SeqFeature
from Bio.SeqFeature import SimpleLocation
from Bio.SeqFeature import LocationParserError
from .utils import FeatureValueCleaner
from .Scanner import GenBankScanner
class _RecordConsumer(_BaseGenBankConsumer):
    """Create a GenBank Record object from scanner generated information (PRIVATE)."""

    def __init__(self):
        _BaseGenBankConsumer.__init__(self)
        from . import Record
        self.data = Record.Record()
        self._seq_data = []
        self._cur_reference = None
        self._cur_feature = None
        self._cur_qualifier = None

    def tls(self, content):
        self.data.tls = content.split('-')

    def tsa(self, content):
        self.data.tsa = content.split('-')

    def wgs(self, content):
        self.data.wgs = content.split('-')

    def add_wgs_scafld(self, content):
        self.data.wgs_scafld.append(content.split('-'))

    def locus(self, content):
        self.data.locus = content

    def size(self, content):
        self.data.size = content

    def residue_type(self, content):
        if 'dna' in content or 'rna' in content:
            warnings.warn(f'Invalid seq_type ({content}): DNA/RNA should be uppercase.', BiopythonParserWarning)
        self.data.residue_type = content

    def data_file_division(self, content):
        self.data.data_file_division = content

    def date(self, content):
        self.data.date = content

    def definition(self, content):
        self.data.definition = content

    def accession(self, content):
        for acc in self._split_accessions(content):
            if acc not in self.data.accession:
                self.data.accession.append(acc)

    def molecule_type(self, mol_type):
        """Validate and record the molecule type (for round-trip etc)."""
        if mol_type:
            if 'circular' in mol_type or 'linear' in mol_type:
                raise ParserFailureError(f'Molecule type {mol_type!r} should not include topology')
            if mol_type[-3:].upper() in ('DNA', 'RNA') and (not mol_type[-3:].isupper()):
                warnings.warn(f'Non-upper case molecule type in LOCUS line: {mol_type}', BiopythonParserWarning)
            self.data.molecule_type = mol_type

    def topology(self, topology):
        """Validate and record sequence topology.

        The topology argument should be "linear" or "circular" (string).
        """
        if topology:
            if topology not in ['linear', 'circular']:
                raise ParserFailureError(f'Unexpected topology {topology!r} should be linear or circular')
            self.data.topology = topology

    def nid(self, content):
        self.data.nid = content

    def pid(self, content):
        self.data.pid = content

    def version(self, content):
        self.data.version = content

    def db_source(self, content):
        self.data.db_source = content.rstrip()

    def gi(self, content):
        self.data.gi = content

    def keywords(self, content):
        self.data.keywords = self._split_keywords(content)

    def project(self, content):
        self.data.projects.extend((p for p in content.split() if p))

    def dblink(self, content):
        self.data.dblinks.append(content)

    def segment(self, content):
        self.data.segment = content

    def source(self, content):
        self.data.source = content

    def organism(self, content):
        self.data.organism = content

    def taxonomy(self, content):
        self.data.taxonomy = self._split_taxonomy(content)

    def reference_num(self, content):
        """Grab the reference number and signal the start of a new reference."""
        if self._cur_reference is not None:
            self.data.references.append(self._cur_reference)
        from . import Record
        self._cur_reference = Record.Reference()
        self._cur_reference.number = content

    def reference_bases(self, content):
        self._cur_reference.bases = content

    def authors(self, content):
        self._cur_reference.authors = content

    def consrtm(self, content):
        self._cur_reference.consrtm = content

    def title(self, content):
        if self._cur_reference is None:
            warnings.warn('GenBank TITLE line without REFERENCE line.', BiopythonParserWarning)
            return
        self._cur_reference.title = content

    def journal(self, content):
        self._cur_reference.journal = content

    def medline_id(self, content):
        self._cur_reference.medline_id = content

    def pubmed_id(self, content):
        self._cur_reference.pubmed_id = content

    def remark(self, content):
        self._cur_reference.remark = content

    def comment(self, content):
        self.data.comment += '\n'.join(content)

    def structured_comment(self, content):
        self.data.structured_comment = content

    def primary_ref_line(self, content):
        """Save reference data for the PRIMARY line."""
        self.data.primary.append(content)

    def primary(self, content):
        pass

    def features_line(self, content):
        """Get ready for the feature table when we reach the FEATURE line."""
        self.start_feature_table()

    def start_feature_table(self):
        """Signal the start of the feature table."""
        if self._cur_reference is not None:
            self.data.references.append(self._cur_reference)

    def feature_key(self, content):
        """Grab the key of the feature and signal the start of a new feature."""
        self._add_feature()
        from . import Record
        self._cur_feature = Record.Feature()
        self._cur_feature.key = content

    def _add_feature(self):
        """Add a feature to the record, with relevant checks (PRIVATE).

        This does all of the appropriate checking to make sure we haven't
        left any info behind, and that we are only adding info if it
        exists.
        """
        if self._cur_feature is not None:
            if self._cur_qualifier is not None:
                self._cur_feature.qualifiers.append(self._cur_qualifier)
            self._cur_qualifier = None
            self.data.features.append(self._cur_feature)

    def location(self, content):
        self._cur_feature.location = self._clean_location(content)

    def feature_qualifier(self, key, value):
        self.feature_qualifier_name([key])
        if value is not None:
            self.feature_qualifier_description(value)

    def feature_qualifier_name(self, content_list):
        """Deal with qualifier names.

        We receive a list of keys, since you can have valueless keys such as
        /pseudo which would be passed in with the next key (since no other
        tags separate them in the file)
        """
        from . import Record
        for content in content_list:
            if not content.startswith('/'):
                content = f'/{content}'
            if self._cur_qualifier is not None:
                self._cur_feature.qualifiers.append(self._cur_qualifier)
            self._cur_qualifier = Record.Qualifier()
            self._cur_qualifier.key = content

    def feature_qualifier_description(self, content):
        if '=' not in self._cur_qualifier.key:
            self._cur_qualifier.key = f'{self._cur_qualifier.key}='
        cur_content = self._remove_newlines(content)
        for remove_space_key in self.__class__.remove_space_keys:
            if remove_space_key in self._cur_qualifier.key:
                cur_content = self._remove_spaces(cur_content)
        self._cur_qualifier.value = self._normalize_spaces(cur_content)

    def base_count(self, content):
        self.data.base_counts = content

    def origin_name(self, content):
        self.data.origin = content

    def contig_location(self, content):
        """Signal that we have contig information to add to the record."""
        self.data.contig = self._clean_location(content)

    def sequence(self, content):
        """Add sequence information to a list of sequence strings.

        This removes spaces in the data and uppercases the sequence, and
        then adds it to a list of sequences. Later on we'll join this
        list together to make the final sequence. This is faster than
        adding on the new string every time.
        """
        assert ' ' not in content
        self._seq_data.append(content.upper())

    def record_end(self, content):
        """Signal the end of the record and do any necessary clean-up."""
        self.data.sequence = ''.join(self._seq_data)
        self._add_feature()