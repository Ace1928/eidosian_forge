import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
def _get_latex_chars_tokens_from_char(self, c: str) -> Tuple[str, Tuple[lexer.Token, ...]]:
    if ord(c) < 128:
        try:
            return self.table.latex_map[c]
        except KeyError:
            pass
    try:
        c.encode(self.inputenc, 'strict')
    except UnicodeEncodeError:
        pass
    else:
        return (c, (lexer.Token(name='chars', text=c),))
    try:
        return self.table.latex_map[c]
    except KeyError:
        if self.errors == 'strict':
            raise UnicodeEncodeError('latex', c, 0, 1, "don't know how to translate {0} into latex".format(repr(c)))
        elif self.errors == 'ignore':
            return ('', (self.emptytoken,))
        elif self.errors == 'replace':
            bytes_ = '{\\char' + str(ord(c)) + '}'
            return (bytes_, (lexer.Token(name='chars', text=bytes_),))
        elif self.errors == 'keep':
            return (c, (lexer.Token(name='chars', text=c),))
        else:
            raise ValueError('latex codec does not support {0} errors'.format(self.errors))