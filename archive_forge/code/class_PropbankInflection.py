import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class PropbankInflection:
    INFINITIVE = 'i'
    GERUND = 'g'
    PARTICIPLE = 'p'
    FINITE = 'v'
    FUTURE = 'f'
    PAST = 'p'
    PRESENT = 'n'
    PERFECT = 'p'
    PROGRESSIVE = 'o'
    PERFECT_AND_PROGRESSIVE = 'b'
    THIRD_PERSON = '3'
    ACTIVE = 'a'
    PASSIVE = 'p'
    NONE = '-'

    def __init__(self, form='-', tense='-', aspect='-', person='-', voice='-'):
        self.form = form
        self.tense = tense
        self.aspect = aspect
        self.person = person
        self.voice = voice

    def __str__(self):
        return self.form + self.tense + self.aspect + self.person + self.voice

    def __repr__(self):
        return '<PropbankInflection: %s>' % self
    _VALIDATE = re.compile('[igpv\\-][fpn\\-][pob\\-][3\\-][ap\\-]$')

    @staticmethod
    def parse(s):
        if not isinstance(s, str):
            raise TypeError('expected a string')
        if len(s) != 5 or not PropbankInflection._VALIDATE.match(s):
            raise ValueError('Bad propbank inflection string %r' % s)
        return PropbankInflection(*s)