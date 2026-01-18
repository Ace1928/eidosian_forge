import re
from nltk.corpus import stopwords
from nltk.stem import porter
from nltk.stem.api import StemmerI
from nltk.stem.util import prefix_replace, suffix_replace
class ArabicStemmer(_StandardStemmer):
    """
    https://github.com/snowballstem/snowball/blob/master/algorithms/arabic/stem_Unicode.sbl (Original Algorithm)
    The Snowball Arabic light Stemmer
    Algorithm:

    - Assem Chelli
    - Abdelkrim Aries
    - Lakhdar Benzahia

    NLTK Version Author:

    - Lakhdar Benzahia
    """
    __vocalization = re.compile('[\\u064b-\\u064c-\\u064d-\\u064e-\\u064f-\\u0650-\\u0651-\\u0652]')
    __kasheeda = re.compile('[\\u0640]')
    __arabic_punctuation_marks = re.compile('[\\u060C-\\u061B-\\u061F]')
    __last_hamzat = ('أ', 'إ', 'آ', 'ؤ', 'ئ')
    __initial_hamzat = re.compile('^[\\u0622\\u0623\\u0625]')
    __waw_hamza = re.compile('[\\u0624]')
    __yeh_hamza = re.compile('[\\u0626]')
    __alefat = re.compile('[\\u0623\\u0622\\u0625]')
    __checks1 = ('كال', 'بال', 'ال', 'لل')
    __checks2 = ('ة', 'ات')
    __suffix_noun_step1a = ('ي', 'ك', 'ه', 'نا', 'كم', 'ها', 'هن', 'هم', 'كما', 'هما')
    __suffix_noun_step1b = 'ن'
    __suffix_noun_step2a = ('ا', 'ي', 'و')
    __suffix_noun_step2b = 'ات'
    __suffix_noun_step2c1 = 'ت'
    __suffix_noun_step2c2 = 'ة'
    __suffix_noun_step3 = 'ي'
    __suffix_verb_step1 = ('ه', 'ك', 'ني', 'نا', 'ها', 'هم', 'هن', 'كم', 'كن', 'هما', 'كما', 'كمو')
    __suffix_verb_step2a = ('ت', 'ا', 'ن', 'ي', 'نا', 'تا', 'تن', 'ان', 'ون', 'ين', 'تما')
    __suffix_verb_step2b = ('وا', 'تم')
    __suffix_verb_step2c = ('و', 'تمو')
    __suffix_all_alef_maqsura = 'ى'
    __prefix_step1 = ('أ', 'أأ', 'أآ', 'أؤ', 'أا', 'أإ')
    __prefix_step2a = ('فال', 'وال')
    __prefix_step2b = ('ف', 'و')
    __prefix_step3a_noun = ('ال', 'لل', 'كال', 'بال')
    __prefix_step3b_noun = ('ب', 'ك', 'ل', 'بب', 'كك')
    __prefix_step3_verb = ('سي', 'ست', 'سن', 'سأ')
    __prefix_step4_verb = ('يست', 'نست', 'تست')
    __conjugation_suffix_verb_1 = ('ه', 'ك')
    __conjugation_suffix_verb_2 = ('ني', 'نا', 'ها', 'هم', 'هن', 'كم', 'كن')
    __conjugation_suffix_verb_3 = ('هما', 'كما', 'كمو')
    __conjugation_suffix_verb_4 = ('ا', 'ن', 'ي')
    __conjugation_suffix_verb_past = ('نا', 'تا', 'تن')
    __conjugation_suffix_verb_present = ('ان', 'ون', 'ين')
    __conjugation_suffix_noun_1 = ('ي', 'ك', 'ه')
    __conjugation_suffix_noun_2 = ('نا', 'كم', 'ها', 'هن', 'هم')
    __conjugation_suffix_noun_3 = ('كما', 'هما')
    __prefixes1 = ('وا', 'فا')
    __articles_3len = ('كال', 'بال')
    __articles_2len = ('ال', 'لل')
    __prepositions1 = ('ك', 'ل')
    __prepositions2 = ('بب', 'كك')
    is_verb = True
    is_noun = True
    is_defined = False
    suffixes_verb_step1_success = False
    suffix_verb_step2a_success = False
    suffix_verb_step2b_success = False
    suffix_noun_step2c2_success = False
    suffix_noun_step1a_success = False
    suffix_noun_step2a_success = False
    suffix_noun_step2b_success = False
    suffixe_noun_step1b_success = False
    prefix_step2a_success = False
    prefix_step3a_noun_success = False
    prefix_step3b_noun_success = False

    def __normalize_pre(self, token):
        """
        :param token: string
        :return: normalized token type string
        """
        token = self.__vocalization.sub('', token)
        token = self.__kasheeda.sub('', token)
        token = self.__arabic_punctuation_marks.sub('', token)
        return token

    def __normalize_post(self, token):
        for hamza in self.__last_hamzat:
            if token.endswith(hamza):
                token = suffix_replace(token, hamza, 'ء')
                break
        token = self.__initial_hamzat.sub('ا', token)
        token = self.__waw_hamza.sub('و', token)
        token = self.__yeh_hamza.sub('ي', token)
        token = self.__alefat.sub('ا', token)
        return token

    def __checks_1(self, token):
        for prefix in self.__checks1:
            if token.startswith(prefix):
                if prefix in self.__articles_3len and len(token) > 4:
                    self.is_noun = True
                    self.is_verb = False
                    self.is_defined = True
                    break
                if prefix in self.__articles_2len and len(token) > 3:
                    self.is_noun = True
                    self.is_verb = False
                    self.is_defined = True
                    break

    def __checks_2(self, token):
        for suffix in self.__checks2:
            if token.endswith(suffix):
                if suffix == 'ة' and len(token) > 2:
                    self.is_noun = True
                    self.is_verb = False
                    break
                if suffix == 'ات' and len(token) > 3:
                    self.is_noun = True
                    self.is_verb = False
                    break

    def __Suffix_Verb_Step1(self, token):
        for suffix in self.__suffix_verb_step1:
            if token.endswith(suffix):
                if suffix in self.__conjugation_suffix_verb_1 and len(token) >= 4:
                    token = token[:-1]
                    self.suffixes_verb_step1_success = True
                    break
                if suffix in self.__conjugation_suffix_verb_2 and len(token) >= 5:
                    token = token[:-2]
                    self.suffixes_verb_step1_success = True
                    break
                if suffix in self.__conjugation_suffix_verb_3 and len(token) >= 6:
                    token = token[:-3]
                    self.suffixes_verb_step1_success = True
                    break
        return token

    def __Suffix_Verb_Step2a(self, token):
        for suffix in self.__suffix_verb_step2a:
            if token.endswith(suffix) and len(token) > 3:
                if suffix == 'ت' and len(token) >= 4:
                    token = token[:-1]
                    self.suffix_verb_step2a_success = True
                    break
                if suffix in self.__conjugation_suffix_verb_4 and len(token) >= 4:
                    token = token[:-1]
                    self.suffix_verb_step2a_success = True
                    break
                if suffix in self.__conjugation_suffix_verb_past and len(token) >= 5:
                    token = token[:-2]
                    self.suffix_verb_step2a_success = True
                    break
                if suffix in self.__conjugation_suffix_verb_present and len(token) > 5:
                    token = token[:-2]
                    self.suffix_verb_step2a_success = True
                    break
                if suffix == 'تما' and len(token) >= 6:
                    token = token[:-3]
                    self.suffix_verb_step2a_success = True
                    break
        return token

    def __Suffix_Verb_Step2c(self, token):
        for suffix in self.__suffix_verb_step2c:
            if token.endswith(suffix):
                if suffix == 'تمو' and len(token) >= 6:
                    token = token[:-3]
                    break
                if suffix == 'و' and len(token) >= 4:
                    token = token[:-1]
                    break
        return token

    def __Suffix_Verb_Step2b(self, token):
        for suffix in self.__suffix_verb_step2b:
            if token.endswith(suffix) and len(token) >= 5:
                token = token[:-2]
                self.suffix_verb_step2b_success = True
                break
        return token

    def __Suffix_Noun_Step2c2(self, token):
        for suffix in self.__suffix_noun_step2c2:
            if token.endswith(suffix) and len(token) >= 3:
                token = token[:-1]
                self.suffix_noun_step2c2_success = True
                break
        return token

    def __Suffix_Noun_Step1a(self, token):
        for suffix in self.__suffix_noun_step1a:
            if token.endswith(suffix):
                if suffix in self.__conjugation_suffix_noun_1 and len(token) >= 4:
                    token = token[:-1]
                    self.suffix_noun_step1a_success = True
                    break
                if suffix in self.__conjugation_suffix_noun_2 and len(token) >= 5:
                    token = token[:-2]
                    self.suffix_noun_step1a_success = True
                    break
                if suffix in self.__conjugation_suffix_noun_3 and len(token) >= 6:
                    token = token[:-3]
                    self.suffix_noun_step1a_success = True
                    break
        return token

    def __Suffix_Noun_Step2a(self, token):
        for suffix in self.__suffix_noun_step2a:
            if token.endswith(suffix) and len(token) > 4:
                token = token[:-1]
                self.suffix_noun_step2a_success = True
                break
        return token

    def __Suffix_Noun_Step2b(self, token):
        for suffix in self.__suffix_noun_step2b:
            if token.endswith(suffix) and len(token) >= 5:
                token = token[:-2]
                self.suffix_noun_step2b_success = True
                break
        return token

    def __Suffix_Noun_Step2c1(self, token):
        for suffix in self.__suffix_noun_step2c1:
            if token.endswith(suffix) and len(token) >= 4:
                token = token[:-1]
                break
        return token

    def __Suffix_Noun_Step1b(self, token):
        for suffix in self.__suffix_noun_step1b:
            if token.endswith(suffix) and len(token) > 5:
                token = token[:-1]
                self.suffixe_noun_step1b_success = True
                break
        return token

    def __Suffix_Noun_Step3(self, token):
        for suffix in self.__suffix_noun_step3:
            if token.endswith(suffix) and len(token) >= 3:
                token = token[:-1]
                break
        return token

    def __Suffix_All_alef_maqsura(self, token):
        for suffix in self.__suffix_all_alef_maqsura:
            if token.endswith(suffix):
                token = suffix_replace(token, suffix, 'ي')
        return token

    def __Prefix_Step1(self, token):
        for prefix in self.__prefix_step1:
            if token.startswith(prefix) and len(token) > 3:
                if prefix == 'أأ':
                    token = prefix_replace(token, prefix, 'أ')
                    break
                elif prefix == 'أآ':
                    token = prefix_replace(token, prefix, 'آ')
                    break
                elif prefix == 'أؤ':
                    token = prefix_replace(token, prefix, 'ؤ')
                    break
                elif prefix == 'أا':
                    token = prefix_replace(token, prefix, 'ا')
                    break
                elif prefix == 'أإ':
                    token = prefix_replace(token, prefix, 'إ')
                    break
        return token

    def __Prefix_Step2a(self, token):
        for prefix in self.__prefix_step2a:
            if token.startswith(prefix) and len(token) > 5:
                token = token[len(prefix):]
                self.prefix_step2a_success = True
                break
        return token

    def __Prefix_Step2b(self, token):
        for prefix in self.__prefix_step2b:
            if token.startswith(prefix) and len(token) > 3:
                if token[:2] not in self.__prefixes1:
                    token = token[len(prefix):]
                    break
        return token

    def __Prefix_Step3a_Noun(self, token):
        for prefix in self.__prefix_step3a_noun:
            if token.startswith(prefix):
                if prefix in self.__articles_2len and len(token) > 4:
                    token = token[len(prefix):]
                    self.prefix_step3a_noun_success = True
                    break
                if prefix in self.__articles_3len and len(token) > 5:
                    token = token[len(prefix):]
                    break
        return token

    def __Prefix_Step3b_Noun(self, token):
        for prefix in self.__prefix_step3b_noun:
            if token.startswith(prefix):
                if len(token) > 3:
                    if prefix == 'ب':
                        token = token[len(prefix):]
                        self.prefix_step3b_noun_success = True
                        break
                    if prefix in self.__prepositions2:
                        token = prefix_replace(token, prefix, prefix[1])
                        self.prefix_step3b_noun_success = True
                        break
                if prefix in self.__prepositions1 and len(token) > 4:
                    token = token[len(prefix):]
                    self.prefix_step3b_noun_success = True
                    break
        return token

    def __Prefix_Step3_Verb(self, token):
        for prefix in self.__prefix_step3_verb:
            if token.startswith(prefix) and len(token) > 4:
                token = prefix_replace(token, prefix, prefix[1])
                break
        return token

    def __Prefix_Step4_Verb(self, token):
        for prefix in self.__prefix_step4_verb:
            if token.startswith(prefix) and len(token) > 4:
                token = prefix_replace(token, prefix, 'است')
                self.is_verb = True
                self.is_noun = False
                break
        return token

    def stem(self, word):
        """
        Stem an Arabic word and return the stemmed form.

        :param word: string
        :return: string
        """
        self.is_verb = True
        self.is_noun = True
        self.is_defined = False
        self.suffix_verb_step2a_success = False
        self.suffix_verb_step2b_success = False
        self.suffix_noun_step2c2_success = False
        self.suffix_noun_step1a_success = False
        self.suffix_noun_step2a_success = False
        self.suffix_noun_step2b_success = False
        self.suffixe_noun_step1b_success = False
        self.prefix_step2a_success = False
        self.prefix_step3a_noun_success = False
        self.prefix_step3b_noun_success = False
        modified_word = word
        self.__checks_1(modified_word)
        self.__checks_2(modified_word)
        modified_word = self.__normalize_pre(modified_word)
        if modified_word in self.stopwords or len(modified_word) <= 2:
            return modified_word
        if self.is_verb:
            modified_word = self.__Suffix_Verb_Step1(modified_word)
            if self.suffixes_verb_step1_success:
                modified_word = self.__Suffix_Verb_Step2a(modified_word)
                if not self.suffix_verb_step2a_success:
                    modified_word = self.__Suffix_Verb_Step2c(modified_word)
            else:
                modified_word = self.__Suffix_Verb_Step2b(modified_word)
                if not self.suffix_verb_step2b_success:
                    modified_word = self.__Suffix_Verb_Step2a(modified_word)
        if self.is_noun:
            modified_word = self.__Suffix_Noun_Step2c2(modified_word)
            if not self.suffix_noun_step2c2_success:
                if not self.is_defined:
                    modified_word = self.__Suffix_Noun_Step1a(modified_word)
                    modified_word = self.__Suffix_Noun_Step2a(modified_word)
                    if not self.suffix_noun_step2a_success:
                        modified_word = self.__Suffix_Noun_Step2b(modified_word)
                    if not self.suffix_noun_step2b_success and (not self.suffix_noun_step2a_success):
                        modified_word = self.__Suffix_Noun_Step2c1(modified_word)
                else:
                    modified_word = self.__Suffix_Noun_Step1b(modified_word)
                    if self.suffixe_noun_step1b_success:
                        modified_word = self.__Suffix_Noun_Step2a(modified_word)
                        if not self.suffix_noun_step2a_success:
                            modified_word = self.__Suffix_Noun_Step2b(modified_word)
                        if not self.suffix_noun_step2b_success and (not self.suffix_noun_step2a_success):
                            modified_word = self.__Suffix_Noun_Step2c1(modified_word)
                    else:
                        if not self.is_defined:
                            modified_word = self.__Suffix_Noun_Step2a(modified_word)
                        modified_word = self.__Suffix_Noun_Step2b(modified_word)
            modified_word = self.__Suffix_Noun_Step3(modified_word)
        if not self.is_noun and self.is_verb:
            modified_word = self.__Suffix_All_alef_maqsura(modified_word)
        modified_word = self.__Prefix_Step1(modified_word)
        modified_word = self.__Prefix_Step2a(modified_word)
        if not self.prefix_step2a_success:
            modified_word = self.__Prefix_Step2b(modified_word)
        modified_word = self.__Prefix_Step3a_Noun(modified_word)
        if not self.prefix_step3a_noun_success and self.is_noun:
            modified_word = self.__Prefix_Step3b_Noun(modified_word)
        elif not self.prefix_step3b_noun_success and self.is_verb:
            modified_word = self.__Prefix_Step3_Verb(modified_word)
            modified_word = self.__Prefix_Step4_Verb(modified_word)
        modified_word = self.__normalize_post(modified_word)
        stemmed_word = modified_word
        return stemmed_word