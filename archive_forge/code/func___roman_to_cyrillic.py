import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
def __roman_to_cyrillic(self, word):
    """
        Transliterate a Russian word back into the Cyrillic alphabet.

        A Russian word formerly transliterated into the Roman alphabet
        in order to ease the stemming process, is transliterated back
        into the Cyrillic alphabet, its original form.

        :param word: The word that is transliterated.
        :type word: str or unicode
        :return: word, the transliterated word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
    word = word.replace('i^u', 'ю').replace('i^a', 'я').replace('shch', 'щ').replace('kh', 'х').replace('t^s', 'ц').replace('ch', 'ч').replace('e`', 'э').replace('i`', 'й').replace('sh', 'ш').replace('k', 'к').replace('e', 'е').replace('zh', 'ж').replace('a', 'а').replace('b', 'б').replace('v', 'в').replace('g', 'г').replace('d', 'д').replace('e', 'е').replace('z', 'з').replace('i', 'и').replace('l', 'л').replace('m', 'м').replace('n', 'н').replace('o', 'о').replace('p', 'п').replace('r', 'р').replace('s', 'с').replace('t', 'т').replace('u', 'у').replace('f', 'ф').replace("''", 'ъ').replace('y', 'ы').replace("'", 'ь')
    return word