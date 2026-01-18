import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
class EmblScanner(InsdcScanner):
    """For extracting chunks of information in EMBL files."""
    RECORD_START = 'ID   '
    HEADER_WIDTH = 5
    FEATURE_START_MARKERS = ['FH   Key             Location/Qualifiers', 'FH']
    FEATURE_END_MARKERS = ['XX']
    FEATURE_QUALIFIER_INDENT = 21
    FEATURE_QUALIFIER_SPACER = 'FT' + ' ' * (FEATURE_QUALIFIER_INDENT - 2)
    SEQUENCE_HEADERS = ['SQ', 'CO']
    EMBL_INDENT = HEADER_WIDTH
    EMBL_SPACER = ' ' * EMBL_INDENT

    def parse_footer(self):
        """Return a tuple containing a list of any misc strings, and the sequence."""
        if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
            raise ValueError(f"Footer format unexpected: '{self.line}'")
        misc_lines = []
        while self.line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
            misc_lines.append(self.line)
            self.line = self.handle.readline()
            if not self.line:
                raise ValueError('Premature end of file')
            self.line = self.line.rstrip()
        if not (self.line[:self.HEADER_WIDTH] == ' ' * self.HEADER_WIDTH or self.line.strip() == '//'):
            raise ValueError(f'Unexpected content after SQ or CO line: {self.line!r}')
        seq_lines = []
        line = self.line
        while True:
            if not line:
                raise ValueError('Premature end of file in sequence data')
            line = line.strip()
            if not line:
                raise ValueError('Blank line in sequence data')
            if line == '//':
                break
            if self.line[:self.HEADER_WIDTH] != ' ' * self.HEADER_WIDTH:
                raise ValueError('Problem with characters in header line,  or incorrect header width: ' + self.line)
            linersplit = line.rsplit(None, 1)
            if len(linersplit) == 2 and linersplit[1].isdigit():
                seq_lines.append(linersplit[0])
            elif line.isdigit():
                pass
            else:
                warnings.warn('EMBL sequence line missing coordinates', BiopythonParserWarning)
                seq_lines.append(line)
            line = self.handle.readline()
        self.line = line
        return (misc_lines, ''.join(seq_lines).replace(' ', ''))

    def _feed_first_line(self, consumer, line):
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        if line[self.HEADER_WIDTH:].count(';') == 6:
            self._feed_first_line_new(consumer, line)
        elif line[self.HEADER_WIDTH:].count(';') == 3:
            if line.rstrip().endswith(' SQ'):
                self._feed_first_line_patents(consumer, line)
            else:
                self._feed_first_line_old(consumer, line)
        elif line[self.HEADER_WIDTH:].count(';') == 2:
            self._feed_first_line_patents_kipo(consumer, line)
        else:
            raise ValueError('Did not recognise the ID line layout:\n' + line)

    def _feed_first_line_patents(self, consumer, line):
        fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip()[:-3].split(';')]
        assert len(fields) == 4
        consumer.locus(fields[0])
        consumer.residue_type(fields[1])
        consumer.data_file_division(fields[2])

    def _feed_first_line_patents_kipo(self, consumer, line):
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        fields = [line[self.HEADER_WIDTH:].split(None, 1)[0]]
        fields.extend(line[self.HEADER_WIDTH:].split(None, 1)[1].split(';'))
        fields = [entry.strip() for entry in fields]
        "\n        The tokens represent:\n\n           0. Primary accession number\n           (space sep)\n           1. ??? (e.g. standard)\n           (semi-colon)\n           2. Molecule type (protein)? Division? Always 'PRT'\n           3. Sequence length (e.g. '111 AA.')\n        "
        consumer.locus(fields[0])
        self._feed_seq_length(consumer, fields[3])

    def _feed_first_line_old(self, consumer, line):
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        fields = [line[self.HEADER_WIDTH:].split(None, 1)[0]]
        fields.extend(line[self.HEADER_WIDTH:].split(None, 1)[1].split(';'))
        fields = [entry.strip() for entry in fields]
        "\n        The tokens represent:\n\n           0. Primary accession number\n           (space sep)\n           1. ??? (e.g. standard)\n           (semi-colon)\n           2. Topology and/or Molecule type (e.g. 'circular DNA' or 'DNA')\n           3. Taxonomic division (e.g. 'PRO')\n           4. Sequence length (e.g. '4639675 BP.')\n\n        "
        consumer.locus(fields[0])
        consumer.residue_type(fields[2])
        if 'circular' in fields[2]:
            consumer.topology('circular')
            consumer.molecule_type(fields[2].replace('circular', '').strip())
        elif 'linear' in fields[2]:
            consumer.topology('linear')
            consumer.molecule_type(fields[2].replace('linear', '').strip())
        else:
            consumer.molecule_type(fields[2].strip())
        consumer.data_file_division(fields[3])
        self._feed_seq_length(consumer, fields[4])

    def _feed_first_line_new(self, consumer, line):
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip().split(';')]
        assert len(fields) == 7
        "\n        The tokens represent:\n\n           0. Primary accession number\n           1. Sequence version number\n           2. Topology: 'circular' or 'linear'\n           3. Molecule type (e.g. 'genomic DNA')\n           4. Data class (e.g. 'STD')\n           5. Taxonomic division (e.g. 'PRO')\n           6. Sequence length (e.g. '4639675 BP.')\n\n        "
        consumer.locus(fields[0])
        consumer.accession(fields[0])
        version_parts = fields[1].split()
        if len(version_parts) == 2 and version_parts[0] == 'SV' and version_parts[1].isdigit():
            consumer.version_suffix(version_parts[1])
        consumer.residue_type(' '.join(fields[2:4]))
        consumer.topology(fields[2])
        consumer.molecule_type(fields[3])
        consumer.data_file_division(fields[5])
        self._feed_seq_length(consumer, fields[6])

    @staticmethod
    def _feed_seq_length(consumer, text):
        length_parts = text.split()
        assert len(length_parts) == 2, f'Invalid sequence length string {text!r}'
        assert length_parts[1].upper() in ['BP', 'BP.', 'AA', 'AA.']
        consumer.size(length_parts[0])

    def _feed_header_lines(self, consumer, lines):
        consumer_dict = {'AC': 'accession', 'SV': 'version', 'DE': 'definition', 'RG': 'consrtm', 'RL': 'journal', 'OS': 'organism', 'OC': 'taxonomy', 'CC': 'comment'}
        for line in lines:
            line_type = line[:self.EMBL_INDENT].strip()
            data = line[self.EMBL_INDENT:].strip()
            if line_type == 'XX':
                pass
            elif line_type == 'RN':
                if data[0] == '[' and data[-1] == ']':
                    data = data[1:-1]
                consumer.reference_num(data)
            elif line_type == 'RP':
                if data.strip() == '[-]':
                    pass
                else:
                    parts = [bases.replace('-', ' to ').strip() for bases in data.split(',') if bases.strip()]
                    consumer.reference_bases(f'(bases {'; '.join(parts)})')
            elif line_type == 'RT':
                if data.startswith('"'):
                    data = data[1:]
                if data.endswith('";'):
                    data = data[:-2]
                consumer.title(data)
            elif line_type == 'RX':
                key, value = data.split(';', 1)
                if value.endswith('.'):
                    value = value[:-1]
                value = value.strip()
                if key == 'PUBMED':
                    consumer.pubmed_id(value)
            elif line_type == 'CC':
                consumer.comment([data])
            elif line_type == 'DR':
                parts = data.rstrip('.').split(';')
                if len(parts) == 1:
                    warnings.warn('Malformed DR line in EMBL file.', BiopythonParserWarning)
                else:
                    consumer.dblink(f'{parts[0].strip()}:{parts[1].strip()}')
            elif line_type == 'RA':
                consumer.authors(data.rstrip(';'))
            elif line_type == 'PR':
                if data.startswith('Project:'):
                    consumer.project(data.rstrip(';'))
            elif line_type == 'KW':
                consumer.keywords(data.rstrip(';'))
            elif line_type in consumer_dict:
                getattr(consumer, consumer_dict[line_type])(data)
            elif self.debug:
                print(f'Ignoring EMBL header line:\n{line}')

    def _feed_misc_lines(self, consumer, lines):
        lines.append('')
        line_iter = iter(lines)
        try:
            for line in line_iter:
                if line.startswith('CO   '):
                    line = line[5:].strip()
                    contig_location = line
                    while True:
                        line = next(line_iter)
                        if not line:
                            break
                        elif line.startswith('CO   '):
                            contig_location += line[5:].strip()
                        else:
                            raise ValueError('Expected CO (contig) continuation line, got:\n' + line)
                    consumer.contig_location(contig_location)
                if line.startswith('SQ   Sequence '):
                    self._feed_seq_length(consumer, line[14:].rstrip().rstrip(';').split(';', 1)[0])
            return
        except StopIteration:
            raise ValueError('Problem in misc lines before sequence') from None