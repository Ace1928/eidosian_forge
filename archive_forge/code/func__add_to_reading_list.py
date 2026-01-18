import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def _add_to_reading_list(self, glueformula, reading_list):
    add_reading = True
    if self.remove_duplicates:
        for reading in reading_list:
            try:
                if reading.equiv(glueformula.meaning, self.prover):
                    add_reading = False
                    break
            except Exception as e:
                print('Error when checking logical equality of statements', e)
    if add_reading:
        reading_list.append(glueformula.meaning)