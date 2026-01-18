from pyparsing import Literal, Word, delimitedList \
def add_fkey_act(toks):
    return ' "%(fromtable)s":%(fromcolumn)s -> "%(totable)s":%(tocolumn)s ' % toks