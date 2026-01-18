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
def _write_feature(self, feature, record_length):
    """Write a single SeqFeature object to features table (PRIVATE)."""
    assert feature.type, feature
    location = _insdc_location_string(feature.location, record_length)
    f_type = feature.type.replace(' ', '_')
    line = (self.QUALIFIER_INDENT_TMP % f_type)[:self.QUALIFIER_INDENT] + self._wrap_location(location) + '\n'
    self.handle.write(line)
    for key, values in feature.qualifiers.items():
        if isinstance(values, (list, tuple)):
            for value in values:
                self._write_feature_qualifier(key, value)
        else:
            self._write_feature_qualifier(key, values)