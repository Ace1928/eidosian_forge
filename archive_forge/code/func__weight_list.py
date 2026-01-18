import sys
from Bio.SeqUtils import ProtParamData  # Local
from Bio.SeqUtils import IsoelectricPoint  # Local
from Bio.Seq import Seq
from Bio.Data import IUPACData
from Bio.SeqUtils import molecular_weight
def _weight_list(self, window, edge):
    """Make list of relative weight of window edges (PRIVATE).

        The relative weight of window edges are compared to the window
        center. The weights are linear. It actually generates half a list.
        For a window of size 9 and edge 0.4 you get a list of
        [0.4, 0.55, 0.7, 0.85].
        """
    unit = 2 * (1.0 - edge) / (window - 1)
    weights = [0.0] * (window // 2)
    for i in range(window // 2):
        weights[i] = edge + unit * i
    return weights