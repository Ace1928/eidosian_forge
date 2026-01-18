import nltk
import os
import re
import itertools
import collections
import pkg_resources
@staticmethod
def _compute_ngrams(evaluated_sentences, reference_sentences, n):
    """
        Computes n-grams overlap of two text collections of sentences.
        Source: http://research.microsoft.com/en-us/um/people/cyl/download/
        papers/rouge-working-note-v1.3.1.pdf

        Args:
          evaluated_sentences: The sentences that have been picked by the
                               summarizer
          reference_sentences: The sentences from the referene set
          n: Size of ngram

        Returns:
          Number of n-grams for evaluated_sentences, reference_sentences and intersection of both.
          intersection of both count multiple of occurences in n-grams match several times

        Raises:
          ValueError: raises exception if a param has len <= 0
        """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError('Collections must contain at least 1 sentence.')
    evaluated_ngrams, _, evaluated_count = Rouge._get_word_ngrams_and_length(n, evaluated_sentences)
    reference_ngrams, _, reference_count = Rouge._get_word_ngrams_and_length(n, reference_sentences)
    overlapping_ngrams = set(evaluated_ngrams.keys()).intersection(set(reference_ngrams.keys()))
    overlapping_count = 0
    for ngram in overlapping_ngrams:
        overlapping_count += min(evaluated_ngrams[ngram], reference_ngrams[ngram])
    return (evaluated_count, reference_count, overlapping_count)