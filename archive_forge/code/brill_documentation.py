from collections import Counter, defaultdict
from nltk import jsontags
from nltk.tag import TaggerI
from nltk.tbl import Feature, Template

        Tags by applying each rule to the entire corpus (rather than all rules to a
        single sequence). The point is to collect statistics on the test set for
        individual rules.

        NOTE: This is inefficient (does not build any index, so will traverse the entire
        corpus N times for N rules) -- usually you would not care about statistics for
        individual rules and thus use batch_tag() instead

        :param sequences: lists of token sequences (sentences, in some applications) to be tagged
        :type sequences: list of list of strings
        :param gold: the gold standard
        :type gold: list of list of strings
        :returns: tuple of (tagged_sequences, ordered list of rule scores (one for each rule))
        