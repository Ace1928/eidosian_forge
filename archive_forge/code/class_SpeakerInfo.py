import sys
import time
from nltk.corpus.reader.api import *
from nltk.internals import import_from_stdlib
from nltk.tree import Tree
class SpeakerInfo:

    def __init__(self, id, sex, dr, use, recdate, birthdate, ht, race, edu, comments=None):
        self.id = id
        self.sex = sex
        self.dr = dr
        self.use = use
        self.recdate = recdate
        self.birthdate = birthdate
        self.ht = ht
        self.race = race
        self.edu = edu
        self.comments = comments

    def __repr__(self):
        attribs = 'id sex dr use recdate birthdate ht race edu comments'
        args = [f'{attr}={getattr(self, attr)!r}' for attr in attribs.split()]
        return 'SpeakerInfo(%s)' % ', '.join(args)