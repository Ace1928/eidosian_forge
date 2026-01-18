import math
import re
from nltk.tokenize.api import TokenizerI
def _normalize_boundaries(self, text, boundaries, paragraph_breaks):
    """Normalize the boundaries identified to the original text's
        paragraph breaks"""
    norm_boundaries = []
    char_count, word_count, gaps_seen = (0, 0, 0)
    seen_word = False
    for char in text:
        char_count += 1
        if char in ' \t\n' and seen_word:
            seen_word = False
            word_count += 1
        if char not in ' \t\n' and (not seen_word):
            seen_word = True
        if gaps_seen < len(boundaries) and word_count > max(gaps_seen * self.w, self.w):
            if boundaries[gaps_seen] == 1:
                best_fit = len(text)
                for br in paragraph_breaks:
                    if best_fit > abs(br - char_count):
                        best_fit = abs(br - char_count)
                        bestbr = br
                    else:
                        break
                if bestbr not in norm_boundaries:
                    norm_boundaries.append(bestbr)
            gaps_seen += 1
    return norm_boundaries