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
class GenBankWriter(_InsdcWriter):
    """GenBank writer."""
    HEADER_WIDTH = 12
    QUALIFIER_INDENT = 21
    STRUCTURED_COMMENT_START = '-START##'
    STRUCTURED_COMMENT_END = '-END##'
    STRUCTURED_COMMENT_DELIM = ' :: '
    LETTERS_PER_LINE = 60
    SEQUENCE_INDENT = 9

    def _write_single_line(self, tag, text):
        """Write single line in each GenBank record (PRIVATE).

        Used in the 'header' of each GenBank record.
        """
        assert len(tag) < self.HEADER_WIDTH
        if len(text) > self.MAX_WIDTH - self.HEADER_WIDTH:
            if tag:
                warnings.warn(f'Annotation {text!r} too long for {tag!r} line', BiopythonWarning)
            else:
                warnings.warn(f'Annotation {text!r} too long', BiopythonWarning)
        self.handle.write('%s%s\n' % (tag.ljust(self.HEADER_WIDTH), text.replace('\n', ' ')))

    def _write_multi_line(self, tag, text):
        """Write multiple lines in each GenBank record (PRIVATE).

        Used in the 'header' of each GenBank record.
        """
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_multi_line(text, max_len)
        self._write_single_line(tag, lines[0])
        for line in lines[1:]:
            self._write_single_line('', line)

    def _write_multi_entries(self, tag, text_list):
        for i, text in enumerate(text_list):
            if i == 0:
                self._write_single_line(tag, text)
            else:
                self._write_single_line('', text)

    @staticmethod
    def _get_date(record):
        default = '01-JAN-1980'
        try:
            date = record.annotations['date']
        except KeyError:
            return default
        if isinstance(date, list) and len(date) == 1:
            date = date[0]
        if isinstance(date, datetime):
            date = date.strftime('%d-%b-%Y').upper()
        months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        if not isinstance(date, str) or len(date) != 11:
            return default
        try:
            datetime(int(date[-4:]), months.index(date[3:6]) + 1, int(date[0:2]))
        except ValueError:
            date = default
        return date

    @staticmethod
    def _get_data_division(record):
        try:
            division = record.annotations['data_file_division']
        except KeyError:
            division = 'UNK'
        if division in ['PRI', 'ROD', 'MAM', 'VRT', 'INV', 'PLN', 'BCT', 'VRL', 'PHG', 'SYN', 'UNA', 'EST', 'PAT', 'STS', 'GSS', 'HTG', 'HTC', 'ENV', 'CON', 'TSA']:
            pass
        else:
            embl_to_gbk = {'FUN': 'PLN', 'HUM': 'PRI', 'MUS': 'ROD', 'PRO': 'BCT', 'UNC': 'UNK', 'XXX': 'UNK'}
            try:
                division = embl_to_gbk[division]
            except KeyError:
                division = 'UNK'
        assert len(division) == 3
        return division

    def _get_topology(self, record):
        """Set the topology to 'circular', 'linear' if defined (PRIVATE)."""
        max_topology_len = len('circular')
        topology = self._get_annotation_str(record, 'topology', default='')
        if topology and len(topology) <= max_topology_len:
            return topology.ljust(max_topology_len)
        else:
            return ' ' * max_topology_len

    def _write_the_first_line(self, record):
        """Write the LOCUS line (PRIVATE)."""
        locus = record.name
        if not locus or locus == '<unknown name>':
            locus = record.id
        if not locus or locus == '<unknown id>':
            locus = self._get_annotation_str(record, 'accession', just_first=True)
        if len(locus) > 16:
            if len(locus) + 1 + len(str(len(record))) > 28:
                warnings.warn('Increasing length of locus line to allow long name. This will result in fields that are not in usual positions.', BiopythonWarning)
        if len(locus.split()) > 1:
            raise ValueError(f'Invalid whitespace in {locus!r} for LOCUS line')
        if len(record) > 99999999999:
            warnings.warn('The sequence length is very long. The LOCUS line will be increased in length to compensate. This may cause unexpected behavior.', BiopythonWarning)
        mol_type = self._get_annotation_str(record, 'molecule_type', None)
        if mol_type is None:
            raise ValueError('missing molecule_type in annotations')
        if mol_type and len(mol_type) > 7:
            mol_type = mol_type.replace('unassigned ', '').replace('genomic ', '')
            if len(mol_type) > 7:
                warnings.warn(f'Molecule type {mol_type!r} too long', BiopythonWarning)
                mol_type = 'DNA'
        if mol_type in ['protein', 'PROTEIN']:
            mol_type = ''
        if mol_type == '':
            units = 'aa'
        else:
            units = 'bp'
        topology = self._get_topology(record)
        division = self._get_data_division(record)
        if len(locus) > 16 and len(str(len(record))) > 11 - (len(locus) - 16):
            name_length = locus + ' ' + str(len(record))
        else:
            name_length = str(len(record)).rjust(28)
            name_length = locus + name_length[len(locus):]
            assert len(name_length) == 28, name_length
            assert ' ' in name_length, name_length
        assert len(units) == 2
        assert len(division) == 3
        line = 'LOCUS       %s %s    %s %s %s %s\n' % (name_length, units, mol_type.ljust(7), topology, division, self._get_date(record))
        if len(line) > 80:
            splitline = line.split()
            if splitline[3] not in ['bp', 'aa']:
                raise ValueError('LOCUS line does not contain size units at expected position:\n' + line)
            if not (splitline[3].strip() == 'aa' or 'DNA' in splitline[4].strip().upper() or 'RNA' in splitline[4].strip().upper()):
                raise ValueError('LOCUS line does not contain valid sequence type (DNA, RNA, ...):\n' + line)
            self.handle.write(line)
        else:
            assert len(line) == 79 + 1, repr(line)
            assert line[12:40].split() == [locus, str(len(record))], line
            if line[40:44] not in [' bp ', ' aa ']:
                raise ValueError('LOCUS line does not contain size units at expected position:\n' + line)
            if line[44:47] not in ['   ', 'ss-', 'ds-', 'ms-']:
                raise ValueError('LOCUS line does not have valid strand type (Single stranded, ...):\n' + line)
            if not (line[47:54].strip() == '' or 'DNA' in line[47:54].strip().upper() or 'RNA' in line[47:54].strip().upper()):
                raise ValueError('LOCUS line does not contain valid sequence type (DNA, RNA, ...):\n' + line)
            if line[54:55] != ' ':
                raise ValueError('LOCUS line does not contain space at position 55:\n' + line)
            if line[55:63].strip() not in ['', 'linear', 'circular']:
                raise ValueError('LOCUS line does not contain valid entry (linear, circular, ...):\n' + line)
            if line[63:64] != ' ':
                raise ValueError('LOCUS line does not contain space at position 64:\n' + line)
            if line[67:68] != ' ':
                raise ValueError('LOCUS line does not contain space at position 68:\n' + line)
            if line[70:71] != '-':
                raise ValueError('LOCUS line does not contain - at position 71 in date:\n' + line)
            if line[74:75] != '-':
                raise ValueError('LOCUS line does not contain - at position 75 in date:\n' + line)
            self.handle.write(line)

    def _write_references(self, record):
        number = 0
        for ref in record.annotations['references']:
            if not isinstance(ref, SeqFeature.Reference):
                continue
            number += 1
            data = str(number)
            if ref.location and len(ref.location) == 1:
                molecule_type = record.annotations.get('molecule_type')
                if molecule_type and 'protein' in molecule_type:
                    units = 'residues'
                else:
                    units = 'bases'
                data += '  (%s %i to %i)' % (units, ref.location[0].start + 1, ref.location[0].end)
            self._write_single_line('REFERENCE', data)
            if ref.authors:
                self._write_multi_line('  AUTHORS', ref.authors)
            if ref.consrtm:
                self._write_multi_line('  CONSRTM', ref.consrtm)
            if ref.title:
                self._write_multi_line('  TITLE', ref.title)
            if ref.journal:
                self._write_multi_line('  JOURNAL', ref.journal)
            if ref.medline_id:
                self._write_multi_line('  MEDLINE', ref.medline_id)
            if ref.pubmed_id:
                self._write_multi_line('   PUBMED', ref.pubmed_id)
            if ref.comment:
                self._write_multi_line('  REMARK', ref.comment)

    def _write_comment(self, record):
        lines = []
        if 'structured_comment' in record.annotations:
            comment = record.annotations['structured_comment']
            padding = 0
            for key, data in comment.items():
                for subkey, subdata in data.items():
                    padding = len(subkey) if len(subkey) > padding else padding
            for key, data in comment.items():
                lines.append(f'##{key}{self.STRUCTURED_COMMENT_START}')
                for subkey, subdata in data.items():
                    spaces = ' ' * (padding - len(subkey))
                    lines.append(f'{subkey}{spaces}{self.STRUCTURED_COMMENT_DELIM}{subdata}')
                lines.append(f'##{key}{self.STRUCTURED_COMMENT_END}')
        if 'comment' in record.annotations:
            comment = record.annotations['comment']
            if isinstance(comment, str):
                lines += comment.split('\n')
            elif isinstance(comment, (list, tuple)):
                lines += list(comment)
            else:
                raise ValueError('Could not understand comment annotation')
        self._write_multi_line('COMMENT', lines[0])
        for line in lines[1:]:
            self._write_multi_line('', line)

    def _write_contig(self, record):
        max_len = self.MAX_WIDTH - self.HEADER_WIDTH
        lines = self._split_contig(record, max_len)
        self._write_single_line('CONTIG', lines[0])
        for text in lines[1:]:
            self._write_single_line('', text)

    def _write_sequence(self, record):
        try:
            data = _get_seq_string(record)
        except UndefinedSequenceError:
            if 'contig' in record.annotations:
                self._write_contig(record)
            else:
                self.handle.write('ORIGIN\n')
            return
        data = data.lower()
        seq_len = len(data)
        self.handle.write('ORIGIN\n')
        for line_number in range(0, seq_len, self.LETTERS_PER_LINE):
            self.handle.write(str(line_number + 1).rjust(self.SEQUENCE_INDENT))
            for words in range(line_number, min(line_number + self.LETTERS_PER_LINE, seq_len), 10):
                self.handle.write(f' {data[words:words + 10]}')
            self.handle.write('\n')

    def write_record(self, record):
        """Write a single record to the output file."""
        handle = self.handle
        self._write_the_first_line(record)
        default = record.id
        if default.count('.') == 1 and default[default.index('.') + 1:].isdigit():
            default = record.id.split('.', 1)[0]
        accession = self._get_annotation_str(record, 'accession', default, just_first=True)
        acc_with_version = accession
        if record.id.startswith(accession + '.'):
            try:
                acc_with_version = '%s.%i' % (accession, int(record.id.split('.', 1)[1]))
            except ValueError:
                pass
        gi = self._get_annotation_str(record, 'gi', just_first=True)
        descr = record.description
        if descr == '<unknown description>':
            descr = ''
        descr += '.'
        self._write_multi_line('DEFINITION', descr)
        self._write_single_line('ACCESSION', accession)
        if gi != '.':
            self._write_single_line('VERSION', f'{acc_with_version}  GI:{gi}')
        else:
            self._write_single_line('VERSION', acc_with_version)
        dbxrefs_with_space = []
        for x in record.dbxrefs:
            if ': ' not in x:
                x = x.replace(':', ': ')
            dbxrefs_with_space.append(x)
        self._write_multi_entries('DBLINK', dbxrefs_with_space)
        del dbxrefs_with_space
        try:
            keywords = '; '.join(record.annotations['keywords'])
            if not keywords.endswith('.'):
                keywords += '.'
        except KeyError:
            keywords = '.'
        self._write_multi_line('KEYWORDS', keywords)
        if 'segment' in record.annotations:
            segment = record.annotations['segment']
            if isinstance(segment, list):
                assert len(segment) == 1, segment
                segment = segment[0]
            self._write_single_line('SEGMENT', segment)
        self._write_multi_line('SOURCE', self._get_annotation_str(record, 'source'))
        org = self._get_annotation_str(record, 'organism')
        if len(org) > self.MAX_WIDTH - self.HEADER_WIDTH:
            org = org[:self.MAX_WIDTH - self.HEADER_WIDTH - 4] + '...'
        self._write_single_line('  ORGANISM', org)
        try:
            taxonomy = '; '.join(record.annotations['taxonomy'])
            if not taxonomy.endswith('.'):
                taxonomy += '.'
        except KeyError:
            taxonomy = '.'
        self._write_multi_line('', taxonomy)
        if 'db_source' in record.annotations:
            db_source = record.annotations['db_source']
            if isinstance(db_source, list):
                db_source = db_source[0]
            self._write_single_line('DBSOURCE', db_source)
        if 'references' in record.annotations:
            self._write_references(record)
        if 'comment' in record.annotations or 'structured_comment' in record.annotations:
            self._write_comment(record)
        handle.write('FEATURES             Location/Qualifiers\n')
        rec_length = len(record)
        for feature in record.features:
            self._write_feature(feature, rec_length)
        self._write_sequence(record)
        handle.write('//\n')