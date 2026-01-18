import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
class _ImgtScanner(EmblScanner):
    """For extracting chunks of information in IMGT (EMBL like) files (PRIVATE).

    IMGT files are like EMBL files but in order to allow longer feature types
    the features should be indented by 25 characters not 21 characters. In
    practice the IMGT flat files tend to use either 21 or 25 characters, so we
    must cope with both.

    This is private to encourage use of Bio.SeqIO rather than Bio.GenBank.
    """
    FEATURE_START_MARKERS = ['FH   Key             Location/Qualifiers', 'FH   Key             Location/Qualifiers (from EMBL)', 'FH   Key                 Location/Qualifiers', 'FH']

    def _feed_first_line(self, consumer, line):
        assert line[:self.HEADER_WIDTH].rstrip() == 'ID'
        if line[self.HEADER_WIDTH:].count(';') != 5:
            return EmblScanner._feed_first_line(self, consumer, line)
        fields = [data.strip() for data in line[self.HEADER_WIDTH:].strip().split(';')]
        assert len(fields) == 6
        "\n        The tokens represent:\n\n           0. Primary accession number (eg 'HLA00001')\n           1. Sequence version number (eg 'SV 1')\n           2. ??? eg 'standard'\n           3. Molecule type (e.g. 'DNA')\n           4. Taxonomic division (e.g. 'HUM')\n           5. Sequence length (e.g. '3503 BP.')\n        "
        consumer.locus(fields[0])
        version_parts = fields[1].split()
        if len(version_parts) == 2 and version_parts[0] == 'SV' and version_parts[1].isdigit():
            consumer.version_suffix(version_parts[1])
        consumer.residue_type(fields[3])
        if 'circular' in fields[3]:
            consumer.topology('circular')
            consumer.molecule_type(fields[3].replace('circular', '').strip())
        elif 'linear' in fields[3]:
            consumer.topology('linear')
            consumer.molecule_type(fields[3].replace('linear', '').strip())
        else:
            consumer.molecule_type(fields[3].strip())
        consumer.data_file_division(fields[4])
        self._feed_seq_length(consumer, fields[5])

    def parse_features(self, skip=False):
        """Return list of tuples for the features (if present).

        Each feature is returned as a tuple (key, location, qualifiers)
        where key and location are strings (e.g. "CDS" and
        "complement(join(490883..490885,1..879))") while qualifiers
        is a list of two string tuples (feature qualifier keys and values).

        Assumes you have already read to the start of the features table.
        """
        if self.line.rstrip() not in self.FEATURE_START_MARKERS:
            if self.debug:
                print("Didn't find any feature table")
            return []
        while self.line.rstrip() in self.FEATURE_START_MARKERS:
            self.line = self.handle.readline()
        bad_position_re = re.compile('([0-9]+)>')
        features = []
        line = self.line
        while True:
            if not line:
                raise ValueError('Premature end of line during features table')
            if line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
                if self.debug:
                    print('Found start of sequence')
                break
            line = line.rstrip()
            if line == '//':
                raise ValueError("Premature end of features table, marker '//' found")
            if line in self.FEATURE_END_MARKERS:
                if self.debug:
                    print('Found end of features')
                line = self.handle.readline()
                break
            if line[2:self.FEATURE_QUALIFIER_INDENT].strip() == '':
                line = self.handle.readline()
                continue
            if skip:
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER:
                    line = self.handle.readline()
            else:
                assert line[:2] == 'FT'
                try:
                    feature_key, location_start = line[2:].strip().split()
                except ValueError:
                    feature_key = line[2:25].strip()
                    location_start = line[25:].strip()
                feature_lines = [location_start]
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER or line.rstrip() == '':
                    assert line[:2] == 'FT'
                    feature_lines.append(line[self.FEATURE_QUALIFIER_INDENT:].strip())
                    line = self.handle.readline()
                feature_key, location, qualifiers = self.parse_feature(feature_key, feature_lines)
                if '>' in location:
                    location = bad_position_re.sub('>\\1', location)
                features.append((feature_key, location, qualifiers))
        self.line = line
        return features