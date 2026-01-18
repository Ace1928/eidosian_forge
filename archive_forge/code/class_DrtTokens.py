import operator
from functools import reduce
from itertools import chain
from nltk.sem.logic import (
class DrtTokens(Tokens):
    DRS = 'DRS'
    DRS_CONC = '+'
    PRONOUN = 'PRO'
    OPEN_BRACKET = '['
    CLOSE_BRACKET = ']'
    COLON = ':'
    PUNCT = [DRS_CONC, OPEN_BRACKET, CLOSE_BRACKET, COLON]
    SYMBOLS = Tokens.SYMBOLS + PUNCT
    TOKENS = Tokens.TOKENS + [DRS] + PUNCT