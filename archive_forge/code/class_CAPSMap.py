from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
class CAPSMap:
    """A map of an alignment showing all possible dcuts.

    Members:
     - alignment - The alignment that is mapped.
     - dcuts - A list of possible CAPS markers in the form of
       DifferentialCutsites.

    """

    def __init__(self, alignment, enzymes=None):
        """Initialize the CAPSMap.

        Required:
         - alignment - The alignment to be mapped.

        Optional:
         - enzymes - List of enzymes to be used to create the map.
           Defaults to an empty list.

        """
        if enzymes is None:
            enzymes = []
        if isinstance(alignment, MultipleSeqAlignment):
            self.sequences = [rec.seq for rec in alignment]
            self.size = len(self.sequences)
            self.length = len(self.sequences[0])
            for seq in self.sequences:
                if len(seq) != self.length:
                    raise AlignmentHasDifferentLengthsError
        else:
            self.sequences = [Seq(s) for s in alignment]
            self.size, self.length = alignment.shape
        self.alignment = alignment
        self.enzymes = enzymes
        self._digest()

    def _digest_with(self, enzyme):
        cuts = []
        all_seq_cuts = []
        for seq in self.sequences:
            seq_cuts = [cut - enzyme.fst5 for cut in enzyme.search(seq)]
            all_seq_cuts.extend(seq_cuts)
            cuts.append(seq_cuts)
        all_seq_cuts = sorted(set(all_seq_cuts))
        for cut in all_seq_cuts:
            cuts_in = []
            blocked_in = []
            for i, seq in enumerate(self.sequences):
                if cut in cuts[i]:
                    cuts_in.append(i)
                else:
                    blocked_in.append(i)
            if cuts_in and blocked_in:
                self.dcuts.append(DifferentialCutsite(start=cut, enzyme=enzyme, cuts_in=cuts_in, blocked_in=blocked_in))

    def _digest(self):
        self.dcuts = []
        for enzyme in self.enzymes:
            self._digest_with(enzyme)