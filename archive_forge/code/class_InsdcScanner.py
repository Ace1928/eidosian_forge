import warnings
import re
import sys
from collections import defaultdict
from Bio.File import as_handle
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import BiopythonParserWarning
from typing import List
class InsdcScanner:
    """Basic functions for breaking up a GenBank/EMBL file into sub sections.

    The International Nucleotide Sequence Database Collaboration (INSDC)
    between the DDBJ, EMBL, and GenBank.  These organisations all use the
    same "Feature Table" layout in their plain text flat file formats.

    However, the header and sequence sections of an EMBL file are very
    different in layout to those produced by GenBank/DDBJ.
    """
    RECORD_START = 'XXX'
    HEADER_WIDTH = 3
    FEATURE_START_MARKERS = ['XXX***FEATURES***XXX']
    FEATURE_END_MARKERS = ['XXX***END FEATURES***XXX']
    FEATURE_QUALIFIER_INDENT = 0
    FEATURE_QUALIFIER_SPACER = ''
    SEQUENCE_HEADERS = ['XXX']

    def __init__(self, debug=0):
        """Initialize the class."""
        assert len(self.RECORD_START) == self.HEADER_WIDTH
        for marker in self.SEQUENCE_HEADERS:
            assert marker == marker.rstrip()
        assert len(self.FEATURE_QUALIFIER_SPACER) == self.FEATURE_QUALIFIER_INDENT
        self.debug = debug
        self.handle = None
        self.line = None

    def set_handle(self, handle):
        """Set the handle attribute."""
        self.handle = handle
        self.line = ''

    def find_start(self):
        """Read in lines until find the ID/LOCUS line, which is returned.

        Any preamble (such as the header used by the NCBI on ``*.seq.gz`` archives)
        will we ignored.
        """
        while True:
            if self.line:
                line = self.line
                self.line = ''
            else:
                line = self.handle.readline()
            if not line:
                if self.debug:
                    print('End of file')
                return None
            if isinstance(line[0], int):
                raise ValueError('Is this handle in binary mode not text mode?')
            if line[:self.HEADER_WIDTH] == self.RECORD_START:
                if self.debug > 1:
                    print('Found the start of a record:\n' + line)
                break
            line = line.rstrip()
            if line == '//':
                if self.debug > 1:
                    print('Skipping // marking end of last record')
            elif line == '':
                if self.debug > 1:
                    print('Skipping blank line before record')
            elif self.debug > 1:
                print('Skipping header line before record:\n' + line)
        self.line = line
        return line

    def parse_header(self):
        """Return list of strings making up the header.

        New line characters are removed.

        Assumes you have just read in the ID/LOCUS line.
        """
        if self.line[:self.HEADER_WIDTH] != self.RECORD_START:
            raise ValueError('Not at start of record')
        header_lines = []
        while True:
            line = self.handle.readline()
            if not line:
                raise ValueError('Premature end of line during sequence data')
            line = line.rstrip()
            if line in self.FEATURE_START_MARKERS:
                if self.debug:
                    print('Found feature table')
                break
            if line[:self.HEADER_WIDTH].rstrip() in self.SEQUENCE_HEADERS:
                if self.debug:
                    print('Found start of sequence')
                break
            if line == '//':
                raise ValueError("Premature end of sequence data marker '//' found")
            header_lines.append(line)
        self.line = line
        return header_lines

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
            if len(line) < self.FEATURE_QUALIFIER_INDENT:
                warnings.warn(f'line too short to contain a feature: {line!r}', BiopythonParserWarning)
                line = self.handle.readline()
                continue
            if skip:
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER:
                    line = self.handle.readline()
            else:
                if line[self.FEATURE_QUALIFIER_INDENT] != ' ' and ' ' in line[self.FEATURE_QUALIFIER_INDENT:]:
                    feature_key, line = line[2:].strip().split(None, 1)
                    feature_lines = [line]
                    warnings.warn(f'Over indented {feature_key} feature?', BiopythonParserWarning)
                else:
                    feature_key = line[2:self.FEATURE_QUALIFIER_INDENT].strip()
                    feature_lines = [line[self.FEATURE_QUALIFIER_INDENT:]]
                line = self.handle.readline()
                while line[:self.FEATURE_QUALIFIER_INDENT] == self.FEATURE_QUALIFIER_SPACER or (line != '' and line.rstrip() == ''):
                    feature_lines.append(line[self.FEATURE_QUALIFIER_INDENT:].strip())
                    line = self.handle.readline()
                features.append(self.parse_feature(feature_key, feature_lines))
        self.line = line
        return features

    def parse_feature(self, feature_key, lines):
        """Parse a feature given as a list of strings into a tuple.

        Expects a feature as a list of strings, returns a tuple (key, location,
        qualifiers)

        For example given this GenBank feature::

             CDS             complement(join(490883..490885,1..879))
                             /locus_tag="NEQ001"
                             /note="conserved hypothetical [Methanococcus jannaschii];
                             COG1583:Uncharacterized ACR; IPR001472:Bipartite nuclear
                             localization signal; IPR002743: Protein of unknown
                             function DUF57"
                             /codon_start=1
                             /transl_table=11
                             /product="hypothetical protein"
                             /protein_id="NP_963295.1"
                             /db_xref="GI:41614797"
                             /db_xref="GeneID:2732620"
                             /translation="MRLLLELKALNSIDKKQLSNYLIQGFIYNILKNTEYSWLHNWKK
                             EKYFNFTLIPKKDIIENKRYYLIISSPDKRFIEVLHNKIKDLDIITIGLAQFQLRKTK
                             KFDPKLRFPWVTITPIVLREGKIVILKGDKYYKVFVKRLEELKKYNLIKKKEPILEEP
                             IEISLNQIKDGWKIIDVKDRYYDFRNKSFSAFSNWLRDLKEQSLRKYNNFCGKNFYFE
                             EAIFEGFTFYKTVSIRIRINRGEAVYIGTLWKELNVYRKLDKEEREFYKFLYDCGLGS
                             LNSMGFGFVNTKKNSAR"

        Then should give input key="CDS" and the rest of the data as a list of strings
        lines=["complement(join(490883..490885,1..879))", ..., "LNSMGFGFVNTKKNSAR"]
        where the leading spaces and trailing newlines have been removed.

        Returns tuple containing: (key as string, location string, qualifiers as list)
        as follows for this example:

        key = "CDS", string
        location = "complement(join(490883..490885,1..879))", string
        qualifiers = list of string tuples:

        [('locus_tag', '"NEQ001"'),
         ('note', '"conserved hypothetical [Methanococcus jannaschii];\\nCOG1583:..."'),
         ('codon_start', '1'),
         ('transl_table', '11'),
         ('product', '"hypothetical protein"'),
         ('protein_id', '"NP_963295.1"'),
         ('db_xref', '"GI:41614797"'),
         ('db_xref', '"GeneID:2732620"'),
         ('translation', '"MRLLLELKALNSIDKKQLSNYLIQGFIYNILKNTEYSWLHNWKK\\nEKYFNFT..."')]

        In the above example, the "note" and "translation" were edited for compactness,
        and they would contain multiple new line characters (displayed above as \\n)

        If a qualifier is quoted (in this case, everything except codon_start and
        transl_table) then the quotes are NOT removed.

        Note that no whitespace is removed.
        """
        iterator = (x for x in lines if x)
        try:
            line = next(iterator)
            feature_location = line.strip()
            while feature_location[-1:] == ',':
                line = next(iterator)
                feature_location += line.strip()
            if feature_location.count('(') > feature_location.count(')'):
                warnings.warn("Non-standard feature line wrapping (didn't break on comma)?", BiopythonParserWarning)
                while feature_location[-1:] == ',' or feature_location.count('(') > feature_location.count(')'):
                    line = next(iterator)
                    feature_location += line.strip()
            qualifiers = []
            for line_number, line in enumerate(iterator):
                if line_number == 0 and line.startswith(')'):
                    feature_location += line.strip()
                elif line[0] == '/':
                    i = line.find('=')
                    key = line[1:i]
                    value = line[i + 1:]
                    if i and value.startswith(' ') and value.lstrip().startswith('"'):
                        warnings.warn('White space after equals in qualifier', BiopythonParserWarning)
                        value = value.lstrip()
                    if i == -1:
                        key = line[1:]
                        qualifiers.append((key, None))
                    elif not value:
                        qualifiers.append((key, ''))
                    elif value == '"':
                        if self.debug:
                            print(f'Single quote {key}:{value}')
                        qualifiers.append((key, value))
                    elif value[0] == '"':
                        value_list = [value]
                        while value_list[-1][-1] != '"':
                            value_list.append(next(iterator))
                        value = '\n'.join(value_list)
                        qualifiers.append((key, value))
                    else:
                        qualifiers.append((key, value))
                else:
                    assert len(qualifiers) > 0
                    assert key == qualifiers[-1][0]
                    if qualifiers[-1][1] is None:
                        raise StopIteration
                    qualifiers[-1] = (key, qualifiers[-1][1] + '\n' + line)
            return (feature_key, feature_location, qualifiers)
        except StopIteration:
            raise ValueError("Problem with '%s' feature:\n%s" % (feature_key, '\n'.join(lines))) from None

    def parse_footer(self):
        """Return a tuple containing a list of any misc strings, and the sequence."""
        if self.line in self.FEATURE_END_MARKERS:
            while self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
                self.line = self.handle.readline()
                if not self.line:
                    raise ValueError('Premature end of file')
                self.line = self.line.rstrip()
        if self.line[:self.HEADER_WIDTH].rstrip() not in self.SEQUENCE_HEADERS:
            raise ValueError('Not at start of sequence')
        while True:
            line = self.handle.readline()
            if not line:
                raise ValueError('Premature end of line during sequence data')
            line = line.rstrip()
            if line == '//':
                break
        self.line = line
        return ([], '')

    def _feed_first_line(self, consumer, line):
        """Handle the LOCUS/ID line, passing data to the consumer (PRIVATE).

        This should be implemented by the EMBL / GenBank specific subclass

        Used by the parse_records() and parse() methods.
        """

    def _feed_header_lines(self, consumer, lines):
        """Handle the header lines (list of strings), passing data to the consumer (PRIVATE).

        This should be implemented by the EMBL / GenBank specific subclass

        Used by the parse_records() and parse() methods.
        """

    @staticmethod
    def _feed_feature_table(consumer, feature_tuples):
        """Handle the feature table (list of tuples), passing data to the consumer (PRIVATE).

        Used by the parse_records() and parse() methods.
        """
        consumer.start_feature_table()
        for feature_key, location_string, qualifiers in feature_tuples:
            consumer.feature_key(feature_key)
            consumer.location(location_string)
            for q_key, q_value in qualifiers:
                if q_value is None:
                    consumer.feature_qualifier(q_key, q_value)
                else:
                    consumer.feature_qualifier(q_key, q_value.replace('\n', ' '))

    def _feed_misc_lines(self, consumer, lines):
        """Handle any lines between features and sequence (list of strings), passing data to the consumer (PRIVATE).

        This should be implemented by the EMBL / GenBank specific subclass

        Used by the parse_records() and parse() methods.
        """

    def feed(self, handle, consumer, do_features=True):
        """Feed a set of data into the consumer.

        This method is intended for use with the "old" code in Bio.GenBank

        Arguments:
         - handle - A handle with the information to parse.
         - consumer - The consumer that should be informed of events.
         - do_features - Boolean, should the features be parsed?
           Skipping the features can be much faster.

        Return values:
         - true  - Passed a record
         - false - Did not find a record

        """
        self.set_handle(handle)
        if not self.find_start():
            consumer.data = None
            return False
        self._feed_first_line(consumer, self.line)
        self._feed_header_lines(consumer, self.parse_header())
        if do_features:
            self._feed_feature_table(consumer, self.parse_features(skip=False))
        else:
            self.parse_features(skip=True)
        misc_lines, sequence_string = self.parse_footer()
        self._feed_misc_lines(consumer, misc_lines)
        consumer.sequence(sequence_string)
        consumer.record_end('//')
        assert self.line == '//'
        return True

    def parse(self, handle, do_features=True):
        """Return a SeqRecord (with SeqFeatures if do_features=True).

        See also the method parse_records() for use on multi-record files.
        """
        from Bio.GenBank import _FeatureConsumer
        from Bio.GenBank.utils import FeatureValueCleaner
        consumer = _FeatureConsumer(use_fuzziness=1, feature_cleaner=FeatureValueCleaner())
        if self.feed(handle, consumer, do_features):
            return consumer.data
        else:
            return None

    def parse_records(self, handle, do_features=True):
        """Parse records, return a SeqRecord object iterator.

        Each record (from the ID/LOCUS line to the // line) becomes a SeqRecord

        The SeqRecord objects include SeqFeatures if do_features=True

        This method is intended for use in Bio.SeqIO
        """
        with as_handle(handle) as handle:
            while True:
                record = self.parse(handle, do_features)
                if record is None:
                    break
                if record.id is None:
                    raise ValueError("Failed to parse the record's ID. Invalid ID line?")
                if record.name == '<unknown name>':
                    raise ValueError("Failed to parse the record's name. Invalid ID line?")
                if record.description == '<unknown description>':
                    raise ValueError("Failed to parse the record's description")
                yield record

    def parse_cds_features(self, handle, alphabet=None, tags2id=('protein_id', 'locus_tag', 'product')):
        """Parse CDS features, return SeqRecord object iterator.

        Each CDS feature becomes a SeqRecord.

        Arguments:
         - alphabet - Obsolete, should be left as None.
         - tags2id  - Tuple of three strings, the feature keys to use
           for the record id, name and description,

        This method is intended for use in Bio.SeqIO

        """
        if alphabet is not None:
            raise ValueError('The alphabet argument is no longer supported')
        with as_handle(handle) as handle:
            self.set_handle(handle)
            while self.find_start():
                self.parse_header()
                feature_tuples = self.parse_features()
                while True:
                    line = self.handle.readline()
                    if not line:
                        break
                    if line[:2] == '//':
                        break
                self.line = line.rstrip()
                for key, location_string, qualifiers in feature_tuples:
                    if key == 'CDS':
                        record = SeqRecord(seq=None)
                        annotations = record.annotations
                        annotations['molecule_type'] = 'protein'
                        annotations['raw_location'] = location_string.replace(' ', '')
                        for qualifier_name, qualifier_data in qualifiers:
                            if qualifier_data is not None and qualifier_data[0] == '"' and (qualifier_data[-1] == '"'):
                                qualifier_data = qualifier_data[1:-1]
                            if qualifier_name == 'translation':
                                assert record.seq is None, 'Multiple translations!'
                                record.seq = Seq(qualifier_data.replace('\n', ''))
                            elif qualifier_name == 'db_xref':
                                record.dbxrefs.append(qualifier_data)
                            else:
                                if qualifier_data is not None:
                                    qualifier_data = qualifier_data.replace('\n', ' ').replace('  ', ' ')
                                try:
                                    annotations[qualifier_name] += ' ' + qualifier_data
                                except KeyError:
                                    annotations[qualifier_name] = qualifier_data
                        try:
                            record.id = annotations[tags2id[0]]
                        except KeyError:
                            pass
                        try:
                            record.name = annotations[tags2id[1]]
                        except KeyError:
                            pass
                        try:
                            record.description = annotations[tags2id[2]]
                        except KeyError:
                            pass
                        yield record