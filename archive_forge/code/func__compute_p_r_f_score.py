import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _compute_p_r_f_score(evaluated_count, reference_count, overlapping_count, alpha=0.5, weight_factor=1.0):
    """
        Compute precision, recall and f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          evaluated_count: #n-grams in the hypothesis
          reference_count: #n-grams in the reference
          overlapping_count: #n-grams in common between hypothesis and reference
          alpha: Value to use for the F1 score (default: 0.5)
          weight_factor: Weight factor if we have use ROUGE-W (default: 1.0, no impact)

        Returns:
          A dict with 'p', 'r' and 'f' as keys fore precision, recall, f1 score
        """
    precision = 0.0 if evaluated_count == 0 else overlapping_count / evaluated_count
    if weight_factor != 1.0:
        precision = precision ** (1.0 / weight_factor)
    recall = 0.0 if reference_count == 0 else overlapping_count / reference_count
    if weight_factor != 1.0:
        recall = recall ** (1.0 / weight_factor)
    f1_score = Rouge._compute_f_score(precision, recall, alpha)
    return {'f': f1_score, 'p': precision, 'r': recall}