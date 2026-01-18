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
class _FeatureConsumer(_BaseGenBankConsumer):
    """Create a SeqRecord object with Features to return (PRIVATE).

    Attributes:
     - use_fuzziness - specify whether or not to parse with fuzziness in
       feature locations.
     - feature_cleaner - a class that will be used to provide specialized
       cleaning-up of feature values.

    """

    def __init__(self, use_fuzziness, feature_cleaner=None):
        from Bio.SeqRecord import SeqRecord
        _BaseGenBankConsumer.__init__(self)
        self.data = SeqRecord(None, id=None)
        self.data.id = None
        self.data.description = ''
        self._use_fuzziness = use_fuzziness
        self._feature_cleaner = feature_cleaner
        self._seq_type = ''
        self._seq_data = []
        self._cur_reference = None
        self._cur_feature = None
        self._expected_size = None

    def locus(self, locus_name):
        """Set the locus name is set as the name of the Sequence."""
        self.data.name = locus_name

    def size(self, content):
        """Record the sequence length."""
        self._expected_size = int(content)

    def residue_type(self, type):
        """Record the sequence type (SEMI-OBSOLETE).

        This reflects the fact that the topology (linear/circular) and
        molecule type (e.g. DNA vs RNA) were a single field in early
        files. Current GenBank/EMBL files have two fields.
        """
        self._seq_type = type.strip()

    def topology(self, topology):
        """Validate and record sequence topology.

        The topology argument should be "linear" or "circular" (string).
        """
        if topology:
            if topology not in ['linear', 'circular']:
                raise ParserFailureError(f'Unexpected topology {topology!r} should be linear or circular')
            self.data.annotations['topology'] = topology

    def molecule_type(self, mol_type):
        """Validate and record the molecule type (for round-trip etc)."""
        if mol_type:
            if 'circular' in mol_type or 'linear' in mol_type:
                raise ParserFailureError(f'Molecule type {mol_type!r} should not include topology')
            if mol_type[-3:].upper() in ('DNA', 'RNA') and (not mol_type[-3:].isupper()):
                warnings.warn(f'Non-upper case molecule type in LOCUS line: {mol_type}', BiopythonParserWarning)
            self.data.annotations['molecule_type'] = mol_type

    def data_file_division(self, division):
        self.data.annotations['data_file_division'] = division

    def date(self, submit_date):
        self.data.annotations['date'] = submit_date

    def definition(self, definition):
        """Set the definition as the description of the sequence."""
        if self.data.description:
            self.data.description += ' ' + definition
        else:
            self.data.description = definition

    def accession(self, acc_num):
        """Set the accession number as the id of the sequence.

        If we have multiple accession numbers, the first one passed is
        used.
        """
        new_acc_nums = self._split_accessions(acc_num)
        try:
            for acc in new_acc_nums:
                if acc not in self.data.annotations['accessions']:
                    self.data.annotations['accessions'].append(acc)
        except KeyError:
            self.data.annotations['accessions'] = new_acc_nums
        if not self.data.id:
            if len(new_acc_nums) > 0:
                self.data.id = self.data.annotations['accessions'][0]

    def tls(self, content):
        self.data.annotations['tls'] = content.split('-')

    def tsa(self, content):
        self.data.annotations['tsa'] = content.split('-')

    def wgs(self, content):
        self.data.annotations['wgs'] = content.split('-')

    def add_wgs_scafld(self, content):
        self.data.annotations.setdefault('wgs_scafld', []).append(content.split('-'))

    def nid(self, content):
        self.data.annotations['nid'] = content

    def pid(self, content):
        self.data.annotations['pid'] = content

    def version(self, version_id):
        if version_id.count('.') == 1 and version_id.split('.')[1].isdigit():
            self.accession(version_id.split('.')[0])
            self.version_suffix(version_id.split('.')[1])
        elif version_id:
            self.data.id = version_id

    def project(self, content):
        """Handle the information from the PROJECT line as a list of projects.

        e.g.::

            PROJECT     GenomeProject:28471

        or::

            PROJECT     GenomeProject:13543  GenomeProject:99999

        This is stored as dbxrefs in the SeqRecord to be consistent with the
        projected switch of this line to DBLINK in future GenBank versions.
        Note the NCBI plan to replace "GenomeProject:28471" with the shorter
        "Project:28471" as part of this transition.
        """
        content = content.replace('GenomeProject:', 'Project:')
        self.data.dbxrefs.extend((p for p in content.split() if p))

    def dblink(self, content):
        """Store DBLINK cross references as dbxrefs in our record object.

        This line type is expected to replace the PROJECT line in 2009. e.g.

        During transition::

            PROJECT     GenomeProject:28471
            DBLINK      Project:28471
                        Trace Assembly Archive:123456

        Once the project line is dropped::

            DBLINK      Project:28471
                        Trace Assembly Archive:123456

        Note GenomeProject -> Project.

        We'll have to see some real examples to be sure, but based on the
        above example we can expect one reference per line.

        Note that at some point the NCBI have included an extra space, e.g.::

            DBLINK      Project: 28471

        """
        while ': ' in content:
            content = content.replace(': ', ':')
        if content.strip() not in self.data.dbxrefs:
            self.data.dbxrefs.append(content.strip())

    def version_suffix(self, version):
        """Set the version to overwrite the id.

        Since the version provides the same information as the accession
        number, plus some extra info, we set this as the id if we have
        a version.
        """
        assert version.isdigit()
        self.data.annotations['sequence_version'] = int(version)

    def db_source(self, content):
        self.data.annotations['db_source'] = content.rstrip()

    def gi(self, content):
        self.data.annotations['gi'] = content

    def keywords(self, content):
        if 'keywords' in self.data.annotations:
            self.data.annotations['keywords'].extend(self._split_keywords(content))
        else:
            self.data.annotations['keywords'] = self._split_keywords(content)

    def segment(self, content):
        self.data.annotations['segment'] = content

    def source(self, content):
        if content == '':
            source_info = ''
        elif content[-1] == '.':
            source_info = content[:-1]
        else:
            source_info = content
        self.data.annotations['source'] = source_info

    def organism(self, content):
        self.data.annotations['organism'] = content

    def taxonomy(self, content):
        """Record (another line of) the taxonomy lineage."""
        lineage = self._split_taxonomy(content)
        try:
            self.data.annotations['taxonomy'].extend(lineage)
        except KeyError:
            self.data.annotations['taxonomy'] = lineage

    def reference_num(self, content):
        """Signal the beginning of a new reference object."""
        if self._cur_reference is not None:
            self.data.annotations['references'].append(self._cur_reference)
        else:
            self.data.annotations['references'] = []
        self._cur_reference = Reference()

    def reference_bases(self, content):
        """Attempt to determine the sequence region the reference entails.

        Possible types of information we may have to deal with:

        (bases 1 to 86436)
        (sites)
        (bases 1 to 105654; 110423 to 111122)
        1  (residues 1 to 182)
        """
        assert content.endswith(')'), content
        ref_base_info = content[1:-1]
        all_locations = []
        if 'bases' in ref_base_info and 'to' in ref_base_info:
            ref_base_info = ref_base_info[5:]
            locations = self._split_reference_locations(ref_base_info)
            all_locations.extend(locations)
        elif 'residues' in ref_base_info and 'to' in ref_base_info:
            residues_start = ref_base_info.find('residues')
            ref_base_info = ref_base_info[residues_start + len('residues '):]
            locations = self._split_reference_locations(ref_base_info)
            all_locations.extend(locations)
        elif ref_base_info == 'sites' or ref_base_info.strip() == 'bases':
            pass
        else:
            raise ValueError(f'Could not parse base info {ref_base_info} in record {self.data.id}')
        self._cur_reference.location = all_locations

    def _split_reference_locations(self, location_string):
        """Get reference locations out of a string of reference information (PRIVATE).

        The passed string should be of the form::

            1 to 20; 20 to 100

        This splits the information out and returns a list of location objects
        based on the reference locations.
        """
        all_base_info = location_string.split(';')
        new_locations = []
        for base_info in all_base_info:
            start, end = base_info.split('to')
            new_start, new_end = self._convert_to_python_numbers(int(start.strip()), int(end.strip()))
            this_location = SimpleLocation(new_start, new_end)
            new_locations.append(this_location)
        return new_locations

    def authors(self, content):
        if self._cur_reference.authors:
            self._cur_reference.authors += ' ' + content
        else:
            self._cur_reference.authors = content

    def consrtm(self, content):
        if self._cur_reference.consrtm:
            self._cur_reference.consrtm += ' ' + content
        else:
            self._cur_reference.consrtm = content

    def title(self, content):
        if self._cur_reference is None:
            warnings.warn('GenBank TITLE line without REFERENCE line.', BiopythonParserWarning)
        elif self._cur_reference.title:
            self._cur_reference.title += ' ' + content
        else:
            self._cur_reference.title = content

    def journal(self, content):
        if self._cur_reference.journal:
            self._cur_reference.journal += ' ' + content
        else:
            self._cur_reference.journal = content

    def medline_id(self, content):
        self._cur_reference.medline_id = content

    def pubmed_id(self, content):
        self._cur_reference.pubmed_id = content

    def remark(self, content):
        """Deal with a reference comment."""
        if self._cur_reference.comment:
            self._cur_reference.comment += ' ' + content
        else:
            self._cur_reference.comment = content

    def comment(self, content):
        try:
            self.data.annotations['comment'] += '\n' + '\n'.join(content)
        except KeyError:
            self.data.annotations['comment'] = '\n'.join(content)

    def structured_comment(self, content):
        self.data.annotations['structured_comment'] = content

    def features_line(self, content):
        """Get ready for the feature table when we reach the FEATURE line."""
        self.start_feature_table()

    def start_feature_table(self):
        """Indicate we've got to the start of the feature table."""
        if self._cur_reference is not None:
            self.data.annotations['references'].append(self._cur_reference)
            self._cur_reference = None

    def feature_key(self, content):
        self._cur_feature = SeqFeature()
        self._cur_feature.type = content
        self.data.features.append(self._cur_feature)

    def location(self, content):
        """Parse out location information from the location string.

        This uses simple Python code with some regular expressions to do the
        parsing, and then translates the results into appropriate objects.
        """
        location_line = self._clean_location(content)
        if 'replace' in location_line:
            comma_pos = location_line.find(',')
            location_line = location_line[8:comma_pos]
        length = self._expected_size
        is_circular = 'circular' in self.data.annotations.get('topology', '').lower()
        stranded = 'PROTEIN' not in self._seq_type.upper()
        try:
            location = Location.fromstring(location_line, length, is_circular, stranded)
        except LocationParserError as e:
            warnings.warn(f'{e}; setting feature location to None.', BiopythonParserWarning)
            location = None
        self._cur_feature.location = location

    def feature_qualifier(self, key, value):
        """When we get a qualifier key and its value.

        Can receive None, since you can have valueless keys such as /pseudo
        """
        if value is None:
            if key not in self._cur_feature.qualifiers:
                self._cur_feature.qualifiers[key] = ['']
                return
            return
        if len(value) > 1 and value[0] == '"' and (value[-1] == '"'):
            value = value[1:-1]
        if re.search('[^"]"[^"]|^"[^"]|[^"]"$', value):
            warnings.warn('The NCBI states double-quote characters like " should be escaped as "" (two double - quotes), but here it was not: %r' % value, BiopythonParserWarning)
        value = value.replace('""', '"')
        if self._feature_cleaner is not None:
            value = self._feature_cleaner.clean_value(key, value)
        if key in self._cur_feature.qualifiers:
            self._cur_feature.qualifiers[key].append(value)
        else:
            self._cur_feature.qualifiers[key] = [value]

    def feature_qualifier_name(self, content_list):
        """Use feature_qualifier instead (OBSOLETE)."""
        raise NotImplementedError('Use the feature_qualifier method instead.')

    def feature_qualifier_description(self, content):
        """Use feature_qualifier instead (OBSOLETE)."""
        raise NotImplementedError('Use the feature_qualifier method instead.')

    def contig_location(self, content):
        """Deal with CONTIG information."""
        self.data.annotations['contig'] = content

    def origin_name(self, content):
        pass

    def base_count(self, content):
        pass

    def base_number(self, content):
        pass

    def sequence(self, content):
        """Add up sequence information as we get it.

        To try and make things speedier, this puts all of the strings
        into a list of strings, and then uses string.join later to put
        them together. Supposedly, this is a big time savings
        """
        assert ' ' not in content
        self._seq_data.append(content.upper())

    def record_end(self, content):
        """Clean up when we've finished the record."""
        if not self.data.id:
            if 'accessions' in self.data.annotations:
                raise ValueError('Problem adding version number to accession: ' + str(self.data.annotations['accessions']))
            self.data.id = self.data.name
        elif self.data.id.count('.') == 0:
            try:
                self.data.id += '.%i' % self.data.annotations['sequence_version']
            except KeyError:
                pass
        sequence = ''.join(self._seq_data)
        if self._expected_size is not None and len(sequence) != 0 and (self._expected_size != len(sequence)):
            warnings.warn('Expected sequence length %i, found %i (%s).' % (self._expected_size, len(sequence), self.data.id), BiopythonParserWarning)
        molecule_type = None
        if self._seq_type:
            if 'DNA' in self._seq_type.upper() or 'MRNA' in self._seq_type.upper():
                molecule_type = 'DNA'
            elif 'RNA' in self._seq_type.upper():
                molecule_type = 'RNA'
            elif 'PROTEIN' in self._seq_type.upper() or self._seq_type == 'PRT':
                molecule_type = 'protein'
            elif self._seq_type in ['circular', 'linear', 'unspecified']:
                pass
            else:
                raise ValueError(f'Could not determine molecule_type for seq_type {self._seq_type}')
        if molecule_type is not None:
            self.data.annotations['molecule_type'] = self.data.annotations.get('molecule_type', molecule_type)
        if not sequence and self._expected_size:
            self.data.seq = Seq(None, length=self._expected_size)
        else:
            self.data.seq = Seq(sequence)