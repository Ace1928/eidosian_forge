from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
class PSIBlast(Header, DatabaseReport, Parameters):
    """Saves the results from a blastpgp search.

    Members:
    rounds       A list of Round objects.
    converged    Whether the search converged.
    + members inherited from base classes

    """

    def __init__(self):
        """Initialize the class."""
        Header.__init__(self)
        DatabaseReport.__init__(self)
        Parameters.__init__(self)
        self.rounds = []
        self.converged = 0