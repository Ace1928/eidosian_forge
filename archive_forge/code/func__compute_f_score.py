import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _compute_f_score(precision, recall, alpha=0.5):
    """
        Compute f1_score (with alpha: P*R / ((1-alpha)*P + alpha*R))

        Args:
          precision: precision
          recall: recall
          overlapping_count: #n-grams in common between hypothesis and reference

        Returns:
            f1 score
        """
    return 0.0 if recall == 0.0 or precision == 0.0 else precision * recall / ((1 - alpha) * precision + alpha * recall)