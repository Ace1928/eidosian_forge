import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _compute_ngrams_lcs(evaluated_sentences, reference_sentences, weight_factor=1.0):
    """
        Computes ROUGE-L (summary level) of two text collections of sentences.
        http://research.microsoft.com/en-us/um/people/cyl/download/papers/
        rouge-working-note-v1.3.1.pdf
        Args:
          evaluated_sentences: The sentences that have been picked by the summarizer
          reference_sentence: One of the sentences in the reference summaries
          weight_factor: Weight factor to be used for WLCS (1.0 by default if LCS)
        Returns:
          Number of LCS n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times
        Raises:
          ValueError: raises exception if a param has len <= 0
        """

    def _lcs(x, y):
        m = len(x)
        n = len(y)
        vals = collections.defaultdict(int)
        dirs = collections.defaultdict(int)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    vals[i, j] = vals[i - 1, j - 1] + 1
                    dirs[i, j] = '|'
                elif vals[i - 1, j] >= vals[i, j - 1]:
                    vals[i, j] = vals[i - 1, j]
                    dirs[i, j] = '^'
                else:
                    vals[i, j] = vals[i, j - 1]
                    dirs[i, j] = '<'
        return (vals, dirs)

    def _wlcs(x, y, weight_factor):
        m = len(x)
        n = len(y)
        vals = collections.defaultdict(float)
        dirs = collections.defaultdict(int)
        lengths = collections.defaultdict(int)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    length_tmp = lengths[i - 1, j - 1]
                    vals[i, j] = vals[i - 1, j - 1] + (length_tmp + 1) ** weight_factor - length_tmp ** weight_factor
                    dirs[i, j] = '|'
                    lengths[i, j] = length_tmp + 1
                elif vals[i - 1, j] >= vals[i, j - 1]:
                    vals[i, j] = vals[i - 1, j]
                    dirs[i, j] = '^'
                    lengths[i, j] = 0
                else:
                    vals[i, j] = vals[i, j - 1]
                    dirs[i, j] = '<'
                    lengths[i, j] = 0
        return (vals, dirs)

    def _mark_lcs(mask, dirs, m, n):
        while m != 0 and n != 0:
            if dirs[m, n] == '|':
                m -= 1
                n -= 1
                mask[m] = 1
            elif dirs[m, n] == '^':
                m -= 1
            elif dirs[m, n] == '<':
                n -= 1
            else:
                raise UnboundLocalError('Illegal move')
        return mask
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError('Collections must contain at least 1 sentence.')
    evaluated_unigrams_dict, evaluated_count = Rouge._get_unigrams(evaluated_sentences)
    reference_unigrams_dict, reference_count = Rouge._get_unigrams(reference_sentences)
    use_WLCS = weight_factor != 1.0
    if use_WLCS:
        evaluated_count = evaluated_count ** weight_factor
        reference_count = 0
    overlapping_count = 0.0
    for reference_sentence in reference_sentences:
        reference_sentence_tokens = reference_sentence.split()
        if use_WLCS:
            reference_count += len(reference_sentence_tokens) ** weight_factor
        hit_mask = [0 for _ in range(len(reference_sentence_tokens))]
        for evaluated_sentence in evaluated_sentences:
            evaluated_sentence_tokens = evaluated_sentence.split()
            if use_WLCS:
                _, lcs_dirs = _wlcs(reference_sentence_tokens, evaluated_sentence_tokens, weight_factor)
            else:
                _, lcs_dirs = _lcs(reference_sentence_tokens, evaluated_sentence_tokens)
            _mark_lcs(hit_mask, lcs_dirs, len(reference_sentence_tokens), len(evaluated_sentence_tokens))
        overlapping_count_length = 0
        for ref_token_id, val in enumerate(hit_mask):
            if val == 1:
                token = reference_sentence_tokens[ref_token_id]
                if evaluated_unigrams_dict[token] > 0 and reference_unigrams_dict[token] > 0:
                    evaluated_unigrams_dict[token] -= 1
                    reference_unigrams_dict[ref_token_id] -= 1
                    if use_WLCS:
                        overlapping_count_length += 1
                        if ref_token_id + 1 < len(hit_mask) and hit_mask[ref_token_id + 1] == 0 or ref_token_id + 1 == len(hit_mask):
                            overlapping_count += overlapping_count_length ** weight_factor
                            overlapping_count_length = 0
                    else:
                        overlapping_count += 1
    if use_WLCS:
        reference_count = reference_count ** weight_factor
    return (evaluated_count, reference_count, overlapping_count)