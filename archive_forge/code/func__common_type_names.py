from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _common_type_names(csource):
    look_for_words = set(COMMON_TYPES)
    look_for_words.add(';')
    look_for_words.add(',')
    look_for_words.add('(')
    look_for_words.add(')')
    look_for_words.add('typedef')
    words_used = set()
    is_typedef = False
    paren = 0
    previous_word = ''
    for word in _r_words.findall(csource):
        if word in look_for_words:
            if word == ';':
                if is_typedef:
                    words_used.discard(previous_word)
                    look_for_words.discard(previous_word)
                    is_typedef = False
            elif word == 'typedef':
                is_typedef = True
                paren = 0
            elif word == '(':
                paren += 1
            elif word == ')':
                paren -= 1
            elif word == ',':
                if is_typedef and paren == 0:
                    words_used.discard(previous_word)
                    look_for_words.discard(previous_word)
            else:
                words_used.add(word)
        previous_word = word
    return words_used