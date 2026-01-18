import warnings
from collections import defaultdict
from nltk.translate import AlignedSent, Alignment, IBMModel, IBMModel1
from nltk.translate.ibm_model import Counts
class Model2Counts(Counts):
    """
    Data object to store counts of various parameters during training.
    Includes counts for alignment.
    """

    def __init__(self):
        super().__init__()
        self.alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0))))
        self.alignment_for_any_i = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0.0)))

    def update_lexical_translation(self, count, s, t):
        self.t_given_s[t][s] += count
        self.any_t_given_s[s] += count

    def update_alignment(self, count, i, j, l, m):
        self.alignment[i][j][l][m] += count
        self.alignment_for_any_i[j][l][m] += count