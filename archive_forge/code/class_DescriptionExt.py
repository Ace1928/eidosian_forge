from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
class DescriptionExt(Description):
    """Extended description record for BLASTXML version 2.

    Members:
    items           List of DescriptionExtItem
    """

    def __init__(self):
        """Initialize the class."""
        super().__init__()
        self.items = []

    def append_item(self, item):
        """Add a description extended record."""
        if len(self.items) == 0:
            self.title = str(item)
        self.items.append(item)