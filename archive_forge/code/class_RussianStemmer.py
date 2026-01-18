import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class RussianStemmer(_LanguageSpecificStemmer):
    """
    The Russian Snowball stemmer.

    :cvar __perfective_gerund_suffixes: Suffixes to be deleted.
    :type __perfective_gerund_suffixes: tuple
    :cvar __adjectival_suffixes: Suffixes to be deleted.
    :type __adjectival_suffixes: tuple
    :cvar __reflexive_suffixes: Suffixes to be deleted.
    :type __reflexive_suffixes: tuple
    :cvar __verb_suffixes: Suffixes to be deleted.
    :type __verb_suffixes: tuple
    :cvar __noun_suffixes: Suffixes to be deleted.
    :type __noun_suffixes: tuple
    :cvar __superlative_suffixes: Suffixes to be deleted.
    :type __superlative_suffixes: tuple
    :cvar __derivational_suffixes: Suffixes to be deleted.
    :type __derivational_suffixes: tuple
    :note: A detailed description of the Russian
           stemming algorithm can be found under
           http://snowball.tartarus.org/algorithms/russian/stemmer.html

    """
    __perfective_gerund_suffixes = ("ivshis'", "yvshis'", "vshis'", 'ivshi', 'yvshi', 'vshi', 'iv', 'yv', 'v')
    __adjectival_suffixes = ('ui^ushchi^ui^u', 'ui^ushchi^ai^a', 'ui^ushchimi', 'ui^ushchymi', 'ui^ushchego', 'ui^ushchogo', 'ui^ushchemu', 'ui^ushchomu', 'ui^ushchikh', 'ui^ushchykh', 'ui^ushchui^u', 'ui^ushchaia', 'ui^ushchoi^u', 'ui^ushchei^u', 'i^ushchi^ui^u', 'i^ushchi^ai^a', 'ui^ushchee', 'ui^ushchie', 'ui^ushchye', 'ui^ushchoe', 'ui^ushchei`', 'ui^ushchii`', 'ui^ushchyi`', 'ui^ushchoi`', 'ui^ushchem', 'ui^ushchim', 'ui^ushchym', 'ui^ushchom', 'i^ushchimi', 'i^ushchymi', 'i^ushchego', 'i^ushchogo', 'i^ushchemu', 'i^ushchomu', 'i^ushchikh', 'i^ushchykh', 'i^ushchui^u', 'i^ushchai^a', 'i^ushchoi^u', 'i^ushchei^u', 'i^ushchee', 'i^ushchie', 'i^ushchye', 'i^ushchoe', 'i^ushchei`', 'i^ushchii`', 'i^ushchyi`', 'i^ushchoi`', 'i^ushchem', 'i^ushchim', 'i^ushchym', 'i^ushchom', 'shchi^ui^u', 'shchi^ai^a', 'ivshi^ui^u', 'ivshi^ai^a', 'yvshi^ui^u', 'yvshi^ai^a', 'shchimi', 'shchymi', 'shchego', 'shchogo', 'shchemu', 'shchomu', 'shchikh', 'shchykh', 'shchui^u', 'shchai^a', 'shchoi^u', 'shchei^u', 'ivshimi', 'ivshymi', 'ivshego', 'ivshogo', 'ivshemu', 'ivshomu', 'ivshikh', 'ivshykh', 'ivshui^u', 'ivshai^a', 'ivshoi^u', 'ivshei^u', 'yvshimi', 'yvshymi', 'yvshego', 'yvshogo', 'yvshemu', 'yvshomu', 'yvshikh', 'yvshykh', 'yvshui^u', 'yvshai^a', 'yvshoi^u', 'yvshei^u', 'vshi^ui^u', 'vshi^ai^a', 'shchee', 'shchie', 'shchye', 'shchoe', 'shchei`', 'shchii`', 'shchyi`', 'shchoi`', 'shchem', 'shchim', 'shchym', 'shchom', 'ivshee', 'ivshie', 'ivshye', 'ivshoe', 'ivshei`', 'ivshii`', 'ivshyi`', 'ivshoi`', 'ivshem', 'ivshim', 'ivshym', 'ivshom', 'yvshee', 'yvshie', 'yvshye', 'yvshoe', 'yvshei`', 'yvshii`', 'yvshyi`', 'yvshoi`', 'yvshem', 'yvshim', 'yvshym', 'yvshom', 'vshimi', 'vshymi', 'vshego', 'vshogo', 'vshemu', 'vshomu', 'vshikh', 'vshykh', 'vshui^u', 'vshai^a', 'vshoi^u', 'vshei^u', 'emi^ui^u', 'emi^ai^a', 'nni^ui^u', 'nni^ai^a', 'vshee', 'vshie', 'vshye', 'vshoe', 'vshei`', 'vshii`', 'vshyi`', 'vshoi`', 'vshem', 'vshim', 'vshym', 'vshom', 'emimi', 'emymi', 'emego', 'emogo', 'ememu', 'emomu', 'emikh', 'emykh', 'emui^u', 'emai^a', 'emoi^u', 'emei^u', 'nnimi', 'nnymi', 'nnego', 'nnogo', 'nnemu', 'nnomu', 'nnikh', 'nnykh', 'nnui^u', 'nnai^a', 'nnoi^u', 'nnei^u', 'emee', 'emie', 'emye', 'emoe', 'emei`', 'emii`', 'emyi`', 'emoi`', 'emem', 'emim', 'emym', 'emom', 'nnee', 'nnie', 'nnye', 'nnoe', 'nnei`', 'nnii`', 'nnyi`', 'nnoi`', 'nnem', 'nnim', 'nnym', 'nnom', 'i^ui^u', 'i^ai^a', 'imi', 'ymi', 'ego', 'ogo', 'emu', 'omu', 'ikh', 'ykh', 'ui^u', 'ai^a', 'oi^u', 'ei^u', 'ee', 'ie', 'ye', 'oe', 'ei`', 'ii`', 'yi`', 'oi`', 'em', 'im', 'ym', 'om')
    __reflexive_suffixes = ('si^a', "s'")
    __verb_suffixes = ("esh'", 'ei`te', 'ui`te', 'ui^ut', "ish'", 'ete', 'i`te', 'i^ut', 'nno', 'ila', 'yla', 'ena', 'ite', 'ili', 'yli', 'ilo', 'ylo', 'eno', 'i^at', 'uet', 'eny', "it'", "yt'", 'ui^u', 'la', 'na', 'li', 'em', 'lo', 'no', 'et', 'ny', "t'", 'ei`', 'ui`', 'il', 'yl', 'im', 'ym', 'en', 'it', 'yt', 'i^u', 'i`', 'l', 'n')
    __noun_suffixes = ('ii^ami', 'ii^akh', 'i^ami', 'ii^am', 'i^akh', 'ami', 'iei`', 'i^am', 'iem', 'akh', 'ii^u', "'i^u", 'ii^a', "'i^a", 'ev', 'ov', 'ie', "'e", 'ei', 'ii', 'ei`', 'oi`', 'ii`', 'em', 'am', 'om', 'i^u', 'i^a', 'a', 'e', 'i', 'i`', 'o', 'u', 'y', "'")
    __superlative_suffixes = ('ei`she', 'ei`sh')
    __derivational_suffixes = ("ost'", 'ost')

    def stem(self, word):
        """
        Stem a Russian word and return the stemmed form.

        :param word: The word that is stemmed.
        :type word: str or unicode
        :return: The stemmed form.
        :rtype: unicode

        """
        if word in self.stopwords:
            return word
        chr_exceeded = False
        for i in range(len(word)):
            if ord(word[i]) > 255:
                chr_exceeded = True
                break
        if not chr_exceeded:
            return word
        word = self.__cyrillic_to_roman(word)
        step1_success = False
        adjectival_removed = False
        verb_removed = False
        undouble_success = False
        superlative_removed = False
        rv, r2 = self.__regions_russian(word)
        for suffix in self.__perfective_gerund_suffixes:
            if rv.endswith(suffix):
                if suffix in ('v', 'vshi', "vshis'"):
                    if rv[-len(suffix) - 3:-len(suffix)] == 'i^a' or rv[-len(suffix) - 1:-len(suffix)] == 'a':
                        word = word[:-len(suffix)]
                        r2 = r2[:-len(suffix)]
                        rv = rv[:-len(suffix)]
                        step1_success = True
                        break
                else:
                    word = word[:-len(suffix)]
                    r2 = r2[:-len(suffix)]
                    rv = rv[:-len(suffix)]
                    step1_success = True
                    break
        if not step1_success:
            for suffix in self.__reflexive_suffixes:
                if rv.endswith(suffix):
                    word = word[:-len(suffix)]
                    r2 = r2[:-len(suffix)]
                    rv = rv[:-len(suffix)]
                    break
            for suffix in self.__adjectival_suffixes:
                if rv.endswith(suffix):
                    if suffix in ('i^ushchi^ui^u', 'i^ushchi^ai^a', 'i^ushchui^u', 'i^ushchai^a', 'i^ushchoi^u', 'i^ushchei^u', 'i^ushchimi', 'i^ushchymi', 'i^ushchego', 'i^ushchogo', 'i^ushchemu', 'i^ushchomu', 'i^ushchikh', 'i^ushchykh', 'shchi^ui^u', 'shchi^ai^a', 'i^ushchee', 'i^ushchie', 'i^ushchye', 'i^ushchoe', 'i^ushchei`', 'i^ushchii`', 'i^ushchyi`', 'i^ushchoi`', 'i^ushchem', 'i^ushchim', 'i^ushchym', 'i^ushchom', 'vshi^ui^u', 'vshi^ai^a', 'shchui^u', 'shchai^a', 'shchoi^u', 'shchei^u', 'emi^ui^u', 'emi^ai^a', 'nni^ui^u', 'nni^ai^a', 'shchimi', 'shchymi', 'shchego', 'shchogo', 'shchemu', 'shchomu', 'shchikh', 'shchykh', 'vshui^u', 'vshai^a', 'vshoi^u', 'vshei^u', 'shchee', 'shchie', 'shchye', 'shchoe', 'shchei`', 'shchii`', 'shchyi`', 'shchoi`', 'shchem', 'shchim', 'shchym', 'shchom', 'vshimi', 'vshymi', 'vshego', 'vshogo', 'vshemu', 'vshomu', 'vshikh', 'vshykh', 'emui^u', 'emai^a', 'emoi^u', 'emei^u', 'nnui^u', 'nnai^a', 'nnoi^u', 'nnei^u', 'vshee', 'vshie', 'vshye', 'vshoe', 'vshei`', 'vshii`', 'vshyi`', 'vshoi`', 'vshem', 'vshim', 'vshym', 'vshom', 'emimi', 'emymi', 'emego', 'emogo', 'ememu', 'emomu', 'emikh', 'emykh', 'nnimi', 'nnymi', 'nnego', 'nnogo', 'nnemu', 'nnomu', 'nnikh', 'nnykh', 'emee', 'emie', 'emye', 'emoe', 'emei`', 'emii`', 'emyi`', 'emoi`', 'emem', 'emim', 'emym', 'emom', 'nnee', 'nnie', 'nnye', 'nnoe', 'nnei`', 'nnii`', 'nnyi`', 'nnoi`', 'nnem', 'nnim', 'nnym', 'nnom'):
                        if rv[-len(suffix) - 3:-len(suffix)] == 'i^a' or rv[-len(suffix) - 1:-len(suffix)] == 'a':
                            word = word[:-len(suffix)]
                            r2 = r2[:-len(suffix)]
                            rv = rv[:-len(suffix)]
                            adjectival_removed = True
                            break
                    else:
                        word = word[:-len(suffix)]
                        r2 = r2[:-len(suffix)]
                        rv = rv[:-len(suffix)]
                        adjectival_removed = True
                        break
            if not adjectival_removed:
                for suffix in self.__verb_suffixes:
                    if rv.endswith(suffix):
                        if suffix in ('la', 'na', 'ete', 'i`te', 'li', 'i`', 'l', 'em', 'n', 'lo', 'no', 'et', 'i^ut', 'ny', "t'", "esh'", 'nno'):
                            if rv[-len(suffix) - 3:-len(suffix)] == 'i^a' or rv[-len(suffix) - 1:-len(suffix)] == 'a':
                                word = word[:-len(suffix)]
                                r2 = r2[:-len(suffix)]
                                rv = rv[:-len(suffix)]
                                verb_removed = True
                                break
                        else:
                            word = word[:-len(suffix)]
                            r2 = r2[:-len(suffix)]
                            rv = rv[:-len(suffix)]
                            verb_removed = True
                            break
            if not adjectival_removed and (not verb_removed):
                for suffix in self.__noun_suffixes:
                    if rv.endswith(suffix):
                        word = word[:-len(suffix)]
                        r2 = r2[:-len(suffix)]
                        rv = rv[:-len(suffix)]
                        break
        if rv.endswith('i'):
            word = word[:-1]
            r2 = r2[:-1]
        for suffix in self.__derivational_suffixes:
            if r2.endswith(suffix):
                word = word[:-len(suffix)]
                break
        if word.endswith('nn'):
            word = word[:-1]
            undouble_success = True
        if not undouble_success:
            for suffix in self.__superlative_suffixes:
                if word.endswith(suffix):
                    word = word[:-len(suffix)]
                    superlative_removed = True
                    break
            if word.endswith('nn'):
                word = word[:-1]
        if not undouble_success and (not superlative_removed):
            if word.endswith("'"):
                word = word[:-1]
        word = self.__roman_to_cyrillic(word)
        return word

    def __regions_russian(self, word):
        """
        Return the regions RV and R2 which are used by the Russian stemmer.

        In any word, RV is the region after the first vowel,
        or the end of the word if it contains no vowel.

        R2 is the region after the first non-vowel following
        a vowel in R1, or the end of the word if there is no such non-vowel.

        R1 is the region after the first non-vowel following a vowel,
        or the end of the word if there is no such non-vowel.

        :param word: The Russian word whose regions RV and R2 are determined.
        :type word: str or unicode
        :return: the regions RV and R2 for the respective Russian word.
        :rtype: tuple
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
        r1 = ''
        r2 = ''
        rv = ''
        vowels = ('A', 'U', 'E', 'a', 'e', 'i', 'o', 'u', 'y')
        word = word.replace('i^a', 'A').replace('i^u', 'U').replace('e`', 'E')
        for i in range(1, len(word)):
            if word[i] not in vowels and word[i - 1] in vowels:
                r1 = word[i + 1:]
                break
        for i in range(1, len(r1)):
            if r1[i] not in vowels and r1[i - 1] in vowels:
                r2 = r1[i + 1:]
                break
        for i in range(len(word)):
            if word[i] in vowels:
                rv = word[i + 1:]
                break
        r2 = r2.replace('A', 'i^a').replace('U', 'i^u').replace('E', 'e`')
        rv = rv.replace('A', 'i^a').replace('U', 'i^u').replace('E', 'e`')
        return (rv, r2)

    def __cyrillic_to_roman(self, word):
        """
        Transliterate a Russian word into the Roman alphabet.

        A Russian word whose letters consist of the Cyrillic
        alphabet are transliterated into the Roman alphabet
        in order to ease the forthcoming stemming process.

        :param word: The word that is transliterated.
        :type word: unicode
        :return: the transliterated word.
        :rtype: unicode
        :note: This helper method is invoked by the stem method of the subclass
               RussianStemmer. It is not to be invoked directly!

        """
        word = word.replace('А', 'a').replace('а', 'a').replace('Б', 'b').replace('б', 'b').replace('В', 'v').replace('в', 'v').replace('Г', 'g').replace('г', 'g').replace('Д', 'd').replace('д', 'd').replace('Е', 'e').replace('е', 'e').replace('Ё', 'e').replace('ё', 'e').replace('Ж', 'zh').replace('ж', 'zh').replace('З', 'z').replace('з', 'z').replace('И', 'i').replace('и', 'i').replace('Й', 'i`').replace('й', 'i`').replace('К', 'k').replace('к', 'k').replace('Л', 'l').replace('л', 'l').replace('М', 'm').replace('м', 'm').replace('Н', 'n').replace('н', 'n').replace('О', 'o').replace('о', 'o').replace('П', 'p').replace('п', 'p').replace('Р', 'r').replace('р', 'r').replace('С', 's').replace('с', 's').replace('Т', 't').replace('т', 't').replace('У', 'u').replace('у', 'u').replace('Ф', 'f').replace('ф', 'f').replace('Х', 'kh').replace('х', 'kh').replace('Ц', 't^s').replace('ц', 't^s').replace('Ч', 'ch').replace('ч', 'ch').replace('Ш', 'sh').replace('ш', 'sh').replace('Щ', 'shch').replace('щ', 'shch').replace('Ъ', "''").replace('ъ', "''").replace('Ы', 'y').replace('ы', 'y').replace('Ь', "'").replace('ь', "'").replace('Э', 'e`').replace('э', 'e`').replace('Ю', 'i^u').replace('ю', 'i^u').replace('Я', 'i^a').replace('я', 'i^a')
        return word

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