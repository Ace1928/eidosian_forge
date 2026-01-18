import warnings
from datetime import datetime
from Bio import BiopythonWarning
from Bio import SeqFeature
from Bio import SeqIO
from Bio.GenBank.Scanner import _ImgtScanner
from Bio.GenBank.Scanner import EmblScanner
from Bio.GenBank.Scanner import GenBankScanner
from Bio.Seq import UndefinedSequenceError
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class EmblWriter(_InsdcWriter):
    """EMBL writer."""
    HEADER_WIDTH = 5
    QUALIFIER_INDENT = 21
    QUALIFIER_INDENT_STR = 'FT' + ' ' * (QUALIFIER_INDENT - 2)
    QUALIFIER_INDENT_TMP = 'FT   %s                '
    FEATURE_HEADER = 'FH   Key             Location/Qualifiers\nFH\n'
    LETTERS_PER_BLOCK = 10
    BLOCKS_PER_LINE = 6
    LETTERS_PER_LINE = LETTERS_PER_BLOCK * BLOCKS_PER_LINE
    POSITION_PADDING = 10

    def _write_contig(self, record):
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_contig(record, max_len)
        for text in lines:
            self._write_single_line('CO', text)

    def _write_sequence(self, record):
        handle = self.handle
        try:
            data = _get_seq_string(record)
        except UndefinedSequenceError:
            if 'contig' in record.annotations:
                self._write_contig(record)
            else:
                handle.write('SQ   \n')
            return
        data = data.lower()
        seq_len = len(data)
        molecule_type = record.annotations.get('molecule_type')
        if molecule_type is not None and 'DNA' in molecule_type:
            a_count = data.count('A') + data.count('a')
            c_count = data.count('C') + data.count('c')
            g_count = data.count('G') + data.count('g')
            t_count = data.count('T') + data.count('t')
            other = seq_len - (a_count + c_count + g_count + t_count)
            handle.write('SQ   Sequence %i BP; %i A; %i C; %i G; %i T; %i other;\n' % (seq_len, a_count, c_count, g_count, t_count, other))
        else:
            handle.write('SQ   \n')
        for line_number in range(seq_len // self.LETTERS_PER_LINE):
            handle.write('    ')
            for block in range(self.BLOCKS_PER_LINE):
                index = self.LETTERS_PER_LINE * line_number + self.LETTERS_PER_BLOCK * block
                handle.write(f' {data[index:index + self.LETTERS_PER_BLOCK]}')
            handle.write(str((line_number + 1) * self.LETTERS_PER_LINE).rjust(self.POSITION_PADDING))
            handle.write('\n')
        if seq_len % self.LETTERS_PER_LINE:
            line_number = seq_len // self.LETTERS_PER_LINE
            handle.write('    ')
            for block in range(self.BLOCKS_PER_LINE):
                index = self.LETTERS_PER_LINE * line_number + self.LETTERS_PER_BLOCK * block
                handle.write(f' {data[index:index + self.LETTERS_PER_BLOCK]}'.ljust(11))
            handle.write(str(seq_len).rjust(self.POSITION_PADDING))
            handle.write('\n')

    def _write_single_line(self, tag, text):
        assert len(tag) == 2
        line = tag + '   ' + text
        if len(text) > self.MAX_WIDTH:
            warnings.warn(f'Line {line!r} too long', BiopythonWarning)
        self.handle.write(line + '\n')

    def _write_multi_line(self, tag, text):
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_multi_line(text, max_len)
        for line in lines:
            self._write_single_line(tag, line)

    def _write_the_first_lines(self, record):
        """Write the ID and AC lines (PRIVATE)."""
        if '.' in record.id and record.id.rsplit('.', 1)[1].isdigit():
            version = 'SV ' + record.id.rsplit('.', 1)[1]
            accession = self._get_annotation_str(record, 'accession', record.id.rsplit('.', 1)[0], just_first=True)
        else:
            version = ''
            accession = self._get_annotation_str(record, 'accession', record.id, just_first=True)
        if ';' in accession:
            raise ValueError(f"Cannot have semi-colon in EMBL accession, '{accession}'")
        if ' ' in accession:
            raise ValueError(f"Cannot have spaces in EMBL accession, '{accession}'")
        topology = self._get_annotation_str(record, 'topology', default='')
        mol_type = record.annotations.get('molecule_type')
        if mol_type is None:
            raise ValueError('missing molecule_type in annotations')
        if mol_type not in ('DNA', 'genomic DNA', 'unassigned DNA', 'mRNA', 'RNA', 'protein'):
            warnings.warn(f'Non-standard molecule type: {mol_type}', BiopythonWarning)
        mol_type_upper = mol_type.upper()
        if 'DNA' in mol_type_upper:
            units = 'BP'
        elif 'RNA' in mol_type_upper:
            units = 'BP'
        elif 'PROTEIN' in mol_type_upper:
            mol_type = 'PROTEIN'
            units = 'AA'
        else:
            raise ValueError(f"failed to understand molecule_type '{mol_type}'")
        division = self._get_data_division(record)
        handle = self.handle
        self._write_single_line('ID', '%s; %s; %s; %s; ; %s; %i %s.' % (accession, version, topology, mol_type, division, len(record), units))
        handle.write('XX\n')
        self._write_single_line('AC', accession + ';')
        handle.write('XX\n')

    @staticmethod
    def _get_data_division(record):
        try:
            division = record.annotations['data_file_division']
        except KeyError:
            division = 'UNC'
        if division in ['PHG', 'ENV', 'FUN', 'HUM', 'INV', 'MAM', 'VRT', 'MUS', 'PLN', 'PRO', 'ROD', 'SYN', 'TGN', 'UNC', 'VRL', 'XXX']:
            pass
        else:
            gbk_to_embl = {'BCT': 'PRO', 'UNK': 'UNC'}
            try:
                division = gbk_to_embl[division]
            except KeyError:
                division = 'UNC'
        assert len(division) == 3
        return division

    def _write_keywords(self, record):
        for keyword in record.annotations['keywords']:
            self._write_single_line('KW', keyword)
        self.handle.write('XX\n')

    def _write_references(self, record):
        number = 0
        for ref in record.annotations['references']:
            if not isinstance(ref, SeqFeature.Reference):
                continue
            number += 1
            self._write_single_line('RN', '[%i]' % number)
            if ref.location and len(ref.location) == 1:
                self._write_single_line('RP', '%i-%i' % (ref.location[0].start + 1, ref.location[0].end))
            if ref.pubmed_id:
                self._write_single_line('RX', f'PUBMED; {ref.pubmed_id}.')
            if ref.consrtm:
                self._write_single_line('RG', f'{ref.consrtm}')
            if ref.authors:
                self._write_multi_line('RA', ref.authors + ';')
            if ref.title:
                self._write_multi_line('RT', f'"{ref.title}";')
            if ref.journal:
                self._write_multi_line('RL', ref.journal)
            self.handle.write('XX\n')

    def _write_comment(self, record):
        comment = record.annotations['comment']
        if isinstance(comment, str):
            lines = comment.split('\n')
        elif isinstance(comment, (list, tuple)):
            lines = comment
        else:
            raise ValueError('Could not understand comment annotation')
        if not lines:
            return
        for line in lines:
            self._write_multi_line('CC', line)
        self.handle.write('XX\n')

    def write_record(self, record):
        """Write a single record to the output file."""
        handle = self.handle
        self._write_the_first_lines(record)
        for xref in sorted(record.dbxrefs):
            if xref.startswith('BioProject:'):
                self._write_single_line('PR', xref[3:] + ';')
                handle.write('XX\n')
                break
            if xref.startswith('Project:'):
                self._write_single_line('PR', xref + ';')
                handle.write('XX\n')
                break
        descr = record.description
        if descr == '<unknown description>':
            descr = '.'
        self._write_multi_line('DE', descr)
        handle.write('XX\n')
        if 'keywords' in record.annotations:
            self._write_keywords(record)
        self._write_multi_line('OS', self._get_annotation_str(record, 'organism'))
        try:
            taxonomy = '; '.join(record.annotations['taxonomy']) + '.'
        except KeyError:
            taxonomy = '.'
        self._write_multi_line('OC', taxonomy)
        handle.write('XX\n')
        if 'references' in record.annotations:
            self._write_references(record)
        if 'comment' in record.annotations:
            self._write_comment(record)
        handle.write(self.FEATURE_HEADER)
        rec_length = len(record)
        for feature in record.features:
            self._write_feature(feature, rec_length)
        handle.write('XX\n')
        self._write_sequence(record)
        handle.write('//\n')