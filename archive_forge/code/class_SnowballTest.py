import unittest
from contextlib import closing
from nltk import data
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
class SnowballTest(unittest.TestCase):

    def test_arabic(self):
        """
        this unit testing for test the snowball arabic light stemmer
        this stemmer deals with prefixes and suffixes
        """
        ar_stemmer = SnowballStemmer('arabic', True)
        assert ar_stemmer.stem('الْعَرَبِــــــيَّة') == 'عرب'
        assert ar_stemmer.stem('العربية') == 'عرب'
        assert ar_stemmer.stem('فقالوا') == 'قال'
        assert ar_stemmer.stem('الطالبات') == 'طالب'
        assert ar_stemmer.stem('فالطالبات') == 'طالب'
        assert ar_stemmer.stem('والطالبات') == 'طالب'
        assert ar_stemmer.stem('الطالبون') == 'طالب'
        assert ar_stemmer.stem('اللذان') == 'اللذان'
        assert ar_stemmer.stem('من') == 'من'
        ar_stemmer = SnowballStemmer('arabic', False)
        assert ar_stemmer.stem('اللذان') == 'اللذ'
        assert ar_stemmer.stem('الطالبات') == 'طالب'
        assert ar_stemmer.stem('الكلمات') == 'كلم'
        ar_stemmer = SnowballStemmer('arabic')
        assert ar_stemmer.stem('الْعَرَبِــــــيَّة') == 'عرب'
        assert ar_stemmer.stem('العربية') == 'عرب'
        assert ar_stemmer.stem('فقالوا') == 'قال'
        assert ar_stemmer.stem('الطالبات') == 'طالب'
        assert ar_stemmer.stem('الكلمات') == 'كلم'

    def test_russian(self):
        stemmer_russian = SnowballStemmer('russian')
        assert stemmer_russian.stem('авантненькая') == 'авантненьк'

    def test_german(self):
        stemmer_german = SnowballStemmer('german')
        stemmer_german2 = SnowballStemmer('german', ignore_stopwords=True)
        assert stemmer_german.stem('Schränke') == 'schrank'
        assert stemmer_german2.stem('Schränke') == 'schrank'
        assert stemmer_german.stem('keinen') == 'kein'
        assert stemmer_german2.stem('keinen') == 'keinen'

    def test_spanish(self):
        stemmer = SnowballStemmer('spanish')
        assert stemmer.stem('Visionado') == 'vision'
        assert stemmer.stem('algue') == 'algu'

    def test_short_strings_bug(self):
        stemmer = SnowballStemmer('english')
        assert stemmer.stem("y's") == 'y'