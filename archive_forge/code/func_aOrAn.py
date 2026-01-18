from pyparsing import *
import random
import string
def aOrAn(item):
    if item.desc[0] in 'aeiou':
        return 'an ' + item.desc
    else:
        return 'a ' + item.desc