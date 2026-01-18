from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from .Interfaces import _clean
from .Interfaces import _get_seq_string
from .Interfaces import SequenceIterator
from .Interfaces import SequenceWriter
class TabWriter(SequenceWriter):
    """Class to write simple tab separated format files.

    Each line consists of "id(tab)sequence" only.

    Any description, name or other annotation is not recorded.

    This class is not intended to be used directly. Instead, please use
    the function ``as_tab``, or the top level ``Bio.SeqIO.write()`` function
    with ``format="tab"``.
    """

    def write_record(self, record):
        """Write a single tab line to the file."""
        assert self._header_written
        assert not self._footer_written
        self._record_written = True
        self.handle.write(as_tab(record))