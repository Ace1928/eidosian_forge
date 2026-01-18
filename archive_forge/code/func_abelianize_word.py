import string
from ..sage_helper import _within_sage, sage_method
def abelianize_word(word, gens):
    return vector(ZZ, [word.count(g) - word.count(g.swapcase()) for g in gens])