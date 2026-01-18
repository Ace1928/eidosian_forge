import re
from nltk.stem.api import StemmerI
def _apply_rule_list(self, word, rules):
    """Applies the first applicable suffix-removal rule to the word

        Takes a word and a list of suffix-removal rules represented as
        3-tuples, with the first element being the suffix to remove,
        the second element being the string to replace it with, and the
        final element being the condition for the rule to be applicable,
        or None if the rule is unconditional.
        """
    for rule in rules:
        suffix, replacement, condition = rule
        if suffix == '*d' and self._ends_double_consonant(word):
            stem = word[:-2]
            if condition is None or condition(stem):
                return stem + replacement
            else:
                return word
        if word.endswith(suffix):
            stem = self._replace_suffix(word, suffix, '')
            if condition is None or condition(stem):
                return stem + replacement
            else:
                return word
    return word