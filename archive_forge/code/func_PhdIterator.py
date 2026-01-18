from the Biopython unit tests:
from typing import Iterator
from Bio.SeqRecord import SeqRecord
from Bio.Sequencing import Phd
from .QualityIO import _get_phred_quality
from .Interfaces import SequenceWriter
from .Interfaces import _TextIOSource
from .Interfaces import _IOSource
def PhdIterator(source: _TextIOSource) -> Iterator[SeqRecord]:
    """Return SeqRecord objects from a PHD file.

    Arguments:
     - source - input stream opened in text mode, or a path to a file

    This uses the Bio.Sequencing.Phd module to do the hard work.
    """
    phd_records = Phd.parse(source)
    for phd_record in phd_records:
        name = phd_record.file_name.split(None, 1)[0]
        seq_record = SeqRecord(phd_record.seq, id=name, name=name, description=phd_record.file_name)
        seq_record.annotations = phd_record.comments
        seq_record.annotations['molecule_type'] = 'DNA'
        seq_record.letter_annotations['phred_quality'] = [int(site[1]) for site in phd_record.sites]
        try:
            seq_record.letter_annotations['peak_location'] = [int(site[2]) for site in phd_record.sites]
        except IndexError:
            pass
        yield seq_record