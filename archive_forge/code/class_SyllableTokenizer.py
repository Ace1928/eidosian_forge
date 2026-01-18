import re
import warnings
from string import punctuation
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams
class SyllableTokenizer(TokenizerI):
    """
    Syllabifies words based on the Sonority Sequencing Principle (SSP).

        >>> from nltk.tokenize import SyllableTokenizer
        >>> from nltk import word_tokenize
        >>> SSP = SyllableTokenizer()
        >>> SSP.tokenize('justification')
        ['jus', 'ti', 'fi', 'ca', 'tion']
        >>> text = "This is a foobar-like sentence."
        >>> [SSP.tokenize(token) for token in word_tokenize(text)]
        [['This'], ['is'], ['a'], ['foo', 'bar', '-', 'li', 'ke'], ['sen', 'ten', 'ce'], ['.']]
    """

    def __init__(self, lang='en', sonority_hierarchy=False):
        """
        :param lang: Language parameter, default is English, 'en'
        :type lang: str
        :param sonority_hierarchy: Sonority hierarchy according to the
                                   Sonority Sequencing Principle.
        :type sonority_hierarchy: list(str)
        """
        if not sonority_hierarchy and lang == 'en':
            sonority_hierarchy = ['aeiouy', 'lmnrw', 'zvsf', 'bcdgtkpqxhj']
        self.vowels = sonority_hierarchy[0]
        self.phoneme_map = {}
        for i, level in enumerate(sonority_hierarchy):
            for c in level:
                sonority_level = len(sonority_hierarchy) - i
                self.phoneme_map[c] = sonority_level
                self.phoneme_map[c.upper()] = sonority_level

    def assign_values(self, token):
        """
        Assigns each phoneme its value from the sonority hierarchy.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return: List of tuples, first element is character/phoneme and
                 second is the soronity value.
        :rtype: list(tuple(str, int))
        """
        syllables_values = []
        for c in token:
            try:
                syllables_values.append((c, self.phoneme_map[c]))
            except KeyError:
                if c not in '0123456789' and c not in punctuation:
                    warnings.warn("Character not defined in sonority_hierarchy, assigning as vowel: '{}'".format(c))
                    syllables_values.append((c, max(self.phoneme_map.values())))
                    if c not in self.vowels:
                        self.vowels += c
                else:
                    syllables_values.append((c, -1))
        return syllables_values

    def validate_syllables(self, syllable_list):
        """
        Ensures each syllable has at least one vowel.
        If the following syllable doesn't have vowel, add it to the current one.

        :param syllable_list: Single word or token broken up into syllables.
        :type syllable_list: list(str)
        :return: Single word or token broken up into syllables
                 (with added syllables if necessary)
        :rtype: list(str)
        """
        valid_syllables = []
        front = ''
        vowel_pattern = re.compile('|'.join(self.vowels))
        for i, syllable in enumerate(syllable_list):
            if syllable in punctuation:
                valid_syllables.append(syllable)
                continue
            if not vowel_pattern.search(syllable):
                if len(valid_syllables) == 0:
                    front += syllable
                else:
                    valid_syllables = valid_syllables[:-1] + [valid_syllables[-1] + syllable]
            elif len(valid_syllables) == 0:
                valid_syllables.append(front + syllable)
            else:
                valid_syllables.append(syllable)
        return valid_syllables

    def tokenize(self, token):
        """
        Apply the SSP to return a list of syllables.
        Note: Sentence/text has to be tokenized first.

        :param token: Single word or token
        :type token: str
        :return syllable_list: Single word or token broken up into syllables.
        :rtype: list(str)
        """
        syllables_values = self.assign_values(token)
        if sum((token.count(x) for x in self.vowels)) <= 1:
            return [token]
        syllable_list = []
        syllable = syllables_values[0][0]
        for trigram in ngrams(syllables_values, n=3):
            phonemes, values = zip(*trigram)
            prev_value, focal_value, next_value = values
            focal_phoneme = phonemes[1]
            if focal_value == -1:
                syllable_list.append(syllable)
                syllable_list.append(focal_phoneme)
                syllable = ''
            elif prev_value >= focal_value == next_value:
                syllable += focal_phoneme
                syllable_list.append(syllable)
                syllable = ''
            elif prev_value > focal_value < next_value:
                syllable_list.append(syllable)
                syllable = ''
                syllable += focal_phoneme
            else:
                syllable += focal_phoneme
        syllable += syllables_values[-1][0]
        syllable_list.append(syllable)
        return self.validate_syllables(syllable_list)