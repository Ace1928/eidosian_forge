from operator import methodcaller
from nltk.lm.api import Smoothing
from nltk.probability import ConditionalFreqDist
def _continuation_counts(self, word, context=tuple()):
    """Count continuations that end with context and word.

        Continuations track unique ngram "types", regardless of how many
        instances were observed for each "type".
        This is different than raw ngram counts which track number of instances.
        """
    higher_order_ngrams_with_context = (counts for prefix_ngram, counts in self.counts[len(context) + 2].items() if prefix_ngram[1:] == context)
    higher_order_ngrams_with_word_count, total = (0, 0)
    for counts in higher_order_ngrams_with_context:
        higher_order_ngrams_with_word_count += int(counts[word] > 0)
        total += _count_values_gt_zero(counts)
    return (higher_order_ngrams_with_word_count, total)