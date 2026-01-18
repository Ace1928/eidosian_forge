import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
class UnicodeLatexIncrementalEncoder(LatexIncrementalEncoder):

    def encode(self, unicode_: str, final: bool=False) -> str:
        return self.uencode(unicode_, final)