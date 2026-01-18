import codecs
import dataclasses
import unicodedata
from typing import Optional, List, Union, Any, Iterator, Tuple, Type, Dict
from latexcodec import lexer
from codecs import CodecInfo
class UnicodeLatexIncrementalDecoder(LatexIncrementalDecoder):

    def decode(self, bytes_: str, final: bool=False) -> str:
        return self.udecode(bytes_, final)