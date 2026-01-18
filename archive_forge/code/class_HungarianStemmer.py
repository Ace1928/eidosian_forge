import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class HungarianStemmer(_LanguageSpecificStemmer):
    """
    The Hungarian Snowball stemmer.

    :cvar __vowels: The Hungarian vowels.
    :type __vowels: unicode
    :cvar __digraphs: The Hungarian digraphs.
    :type __digraphs: tuple
    :cvar __double_consonants: The Hungarian double consonants.
    :type __double_consonants: tuple
    :cvar __step1_suffixes: Suffixes to be deleted in step 1 of the algorithm.
    :type __step1_suffixes: tuple
    :cvar __step2_suffixes: Suffixes to be deleted in step 2 of the algorithm.
    :type __step2_suffixes: tuple
    :cvar __step3_suffixes: Suffixes to be deleted in step 3 of the algorithm.
    :type __step3_suffixes: tuple
    :cvar __step4_suffixes: Suffixes to be deleted in step 4 of the algorithm.
    :type __step4_suffixes: tuple
    :cvar __step5_suffixes: Suffixes to be deleted in step 5 of the algorithm.
    :type __step5_suffixes: tuple
    :cvar __step6_suffixes: Suffixes to be deleted in step 6 of the algorithm.
    :type __step6_suffixes: tuple
    :cvar __step7_suffixes: Suffixes to be deleted in step 7 of the algorithm.
    :type __step7_suffixes: tuple
    :cvar __step8_suffixes: Suffixes to be deleted in step 8 of the algorithm.
    :type __step8_suffixes: tuple
    :cvar __step9_suffixes: Suffixes to be deleted in step 9 of the algorithm.
    :type __step9_suffixes: tuple
    :note: A detailed description of the Hungarian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/hungarian/stemmer.html

    """
    __vowels = 'aeiouöüáéíóõúû'
    __digraphs = ('cs', 'dz', 'dzs', 'gy', 'ly', 'ny', 'ty', 'zs')
    __double_consonants = ('bb', 'cc', 'ccs', 'dd', 'ff', 'gg', 'ggy', 'jj', 'kk', 'll', 'lly', 'mm', 'nn', 'nny', 'pp', 'rr', 'ss', 'ssz', 'tt', 'tty', 'vv', 'zz', 'zzs')
    __step1_suffixes = ('al', 'el')
    __step2_suffixes = ('képpen', 'onként', 'enként', 'anként', 'képp', 'ként', 'ban', 'ben', 'nak', 'nek', 'val', 'vel', 'tól', 'tõl', 'ról', 'rõl', 'ból', 'bõl', 'hoz', 'hez', 'höz', 'nál', 'nél', 'ért', 'kor', 'ba', 'be', 'ra', 're', 'ig', 'at', 'et', 'ot', 'öt', 'ul', 'ül', 'vá', 'vé', 'en', 'on', 'an', 'ön', 'n', 't')
    __step3_suffixes = ('ánként', 'án', 'én')
    __step4_suffixes = ('astul', 'estül', 'ástul', 'éstül', 'stul', 'stül')
    __step5_suffixes = ('á', 'é')
    __step6_suffixes = ('oké', 'öké', 'aké', 'eké', 'áké', 'áéi', 'éké', 'ééi', 'ké', 'éi', 'éé', 'é')
    __step7_suffixes = ('ájuk', 'éjük', 'ünk', 'unk', 'juk', 'jük', 'ánk', 'énk', 'nk', 'uk', 'ük', 'em', 'om', 'am', 'od', 'ed', 'ad', 'öd', 'ja', 'je', 'ám', 'ád', 'ém', 'éd', 'm', 'd', 'a', 'e', 'o', 'á', 'é')
    __step8_suffixes = ('jaitok', 'jeitek', 'jaink', 'jeink', 'aitok', 'eitek', 'áitok', 'éitek', 'jaim', 'jeim', 'jaid', 'jeid', 'eink', 'aink', 'itek', 'jeik', 'jaik', 'áink', 'éink', 'aim', 'eim', 'aid', 'eid', 'jai', 'jei', 'ink', 'aik', 'eik', 'áim', 'áid', 'áik', 'éim', 'éid', 'éik', 'im', 'id', 'ai', 'ei', 'ik', 'ái', 'éi', 'i')
    __step9_suffixes = ('ák', 'ék', 'ök', 'ok', 'ek', 'ak', 'k')

    def stem(self, word):
        """
        Stem an Hungarian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        word = word.lower()
        if word in self.stopwords:
            return word
        r1 = self.__r1_hungarian(word, self.__vowels, self.__digraphs)
        if r1.endswith(self.__step1_suffixes):
            for double_cons in self.__double_consonants:
                if word[-2 - len(double_cons):-2] == double_cons:
                    word = ''.join((word[:-4], word[-3]))
                    if r1[-2 - len(double_cons):-2] == double_cons:
                        r1 = ''.join((r1[:-4], r1[-3]))
                    break
        for suffix in self.__step2_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                    if r1.endswith('á'):
                        word = ''.join((word[:-1], 'a'))
                        r1 = suffix_replace(r1, 'á', 'a')
                    elif r1.endswith('é'):
                        word = ''.join((word[:-1], 'e'))
                        r1 = suffix_replace(r1, 'é', 'e')
                break
        for suffix in self.__step3_suffixes:
            if r1.endswith(suffix):
                if suffix == 'én':
                    word = suffix_replace(word, suffix, 'e')
                    r1 = suffix_replace(r1, suffix, 'e')
                else:
                    word = suffix_replace(word, suffix, 'a')
                    r1 = suffix_replace(r1, suffix, 'a')
                break
        for suffix in self.__step4_suffixes:
            if r1.endswith(suffix):
                if suffix == 'ástul':
                    word = suffix_replace(word, suffix, 'a')
                    r1 = suffix_replace(r1, suffix, 'a')
                elif suffix == 'éstül':
                    word = suffix_replace(word, suffix, 'e')
                    r1 = suffix_replace(r1, suffix, 'e')
                else:
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                break
        for suffix in self.__step5_suffixes:
            if r1.endswith(suffix):
                for double_cons in self.__double_consonants:
                    if word[-1 - len(double_cons):-1] == double_cons:
                        word = ''.join((word[:-3], word[-2]))
                        if r1[-1 - len(double_cons):-1] == double_cons:
                            r1 = ''.join((r1[:-3], r1[-2]))
                        break
        for suffix in self.__step6_suffixes:
            if r1.endswith(suffix):
                if suffix in ('áké', 'áéi'):
                    word = suffix_replace(word, suffix, 'a')
                    r1 = suffix_replace(r1, suffix, 'a')
                elif suffix in ('éké', 'ééi', 'éé'):
                    word = suffix_replace(word, suffix, 'e')
                    r1 = suffix_replace(r1, suffix, 'e')
                else:
                    word = word[:-len(suffix)]
                    r1 = r1[:-len(suffix)]
                break
        for suffix in self.__step7_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix in ('ánk', 'ájuk', 'ám', 'ád', 'á'):
                        word = suffix_replace(word, suffix, 'a')
                        r1 = suffix_replace(r1, suffix, 'a')
                    elif suffix in ('énk', 'éjük', 'ém', 'éd', 'é'):
                        word = suffix_replace(word, suffix, 'e')
                        r1 = suffix_replace(r1, suffix, 'e')
                    else:
                        word = word[:-len(suffix)]
                        r1 = r1[:-len(suffix)]
                break
        for suffix in self.__step8_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix in ('áim', 'áid', 'ái', 'áink', 'áitok', 'áik'):
                        word = suffix_replace(word, suffix, 'a')
                        r1 = suffix_replace(r1, suffix, 'a')
                    elif suffix in ('éim', 'éid', 'éi', 'éink', 'éitek', 'éik'):
                        word = suffix_replace(word, suffix, 'e')
                        r1 = suffix_replace(r1, suffix, 'e')
                    else:
                        word = word[:-len(suffix)]
                        r1 = r1[:-len(suffix)]
                break
        for suffix in self.__step9_suffixes:
            if word.endswith(suffix):
                if r1.endswith(suffix):
                    if suffix == 'ák':
                        word = suffix_replace(word, suffix, 'a')
                    elif suffix == 'ék':
                        word = suffix_replace(word, suffix, 'e')
                    else:
                        word = word[:-len(suffix)]
                break
        return word

    def __r1_hungarian(self, word, vowels, digraphs):
        """
        Return the region R1 that is used by the Hungarian stemmer.

        If the word begins with a vowel, R1 is defined as the region
        after the first consonant or digraph (= two letters stand for
        one phoneme) in the word. If the word begins with a consonant,
        it is defined as the region after the first vowel in the word.
        If the word does not contain both a vowel and consonant, R1
        is the null region at the end of the word.

        :param word: The Hungarian word whose region R1 is determined.
        :type word: str or unicode
        :param vowels: The Hungarian vowels that are used to determine
                       the region R1.
        :type vowels: unicode
        :param digraphs: The digraphs that are used to determine the
                         region R1.
        :type digraphs: tuple
        :return: the region R1 for the respective word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               HungarianStemmer. It is not to be invoked directly!

        """
        r1 = ''
        if word[0] in vowels:
            for digraph in digraphs:
                if digraph in word[1:]:
                    r1 = word[word.index(digraph[-1]) + 1:]
                    return r1
            for i in range(1, len(word)):
                if word[i] not in vowels:
                    r1 = word[i + 1:]
                    break
        else:
            for i in range(1, len(word)):
                if word[i] in vowels:
                    r1 = word[i + 1:]
                    break
        return r1