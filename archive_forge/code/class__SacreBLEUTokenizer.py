import os
import re
import tempfile
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Sequence, Type
import torch
from torch import Tensor, tensor
from typing_extensions import Literal
from torchmetrics.functional.text.bleu import _bleu_score_compute, _bleu_score_update
from torchmetrics.utilities.imports import (
class _SacreBLEUTokenizer:
    """Tokenizer used for SacreBLEU calculation.

    Source: https://github.com/mjpost/sacrebleu/tree/master/sacrebleu/tokenizers

    """
    _REGEX = ((re.compile('([\\{-\\~\\[-\\` -\\&\\(-\\+\\:-\\@\\/])'), ' \\1 '), (re.compile('([^0-9])([\\.,])'), '\\1 \\2 '), (re.compile('([\\.,])([^0-9])'), ' \\1 \\2'), (re.compile('([0-9])(-)'), '\\1 \\2 '))
    if _REGEX_AVAILABLE:
        import regex
        _INT_REGEX = ((regex.compile('(\\P{N})(\\p{P})'), '\\1 \\2 '), (regex.compile('(\\p{P})(\\P{N})'), ' \\1 \\2'), (regex.compile('(\\p{S})'), ' \\1 '))
    _TOKENIZE_FN: ClassVar[dict] = {'none': '_tokenize_base', '13a': '_tokenize_13a', 'zh': '_tokenize_zh', 'intl': '_tokenize_international', 'char': '_tokenize_char', 'ja-mecab': '_tokenize_ja_mecab', 'ko-mecab': '_tokenize_ko_mecab', 'flores101': '_tokenize_flores_101', 'flores200': '_tokenize_flores_200'}
    sentencepiece_processors: ClassVar[Dict[str, Optional[Any]]] = {'flores101': None, 'flores200': None}

    def __init__(self, tokenize: _TokenizersLiteral, lowercase: bool=False) -> None:
        self._check_tokenizers_validity(tokenize)
        self.tokenize_fn = getattr(self, self._TOKENIZE_FN[tokenize])
        self.lowercase = lowercase

    def __call__(self, line: str) -> Sequence[str]:
        tokenized_line = self.tokenize_fn(line)
        return self._lower(tokenized_line, self.lowercase).split()

    @classmethod
    def tokenize(cls: Type['_SacreBLEUTokenizer'], line: str, tokenize: _TokenizersLiteral, lowercase: bool=False) -> Sequence[str]:
        cls._check_tokenizers_validity(tokenize)
        tokenize_fn = getattr(cls, cls._TOKENIZE_FN[tokenize])
        tokenized_line = tokenize_fn(line)
        return cls._lower(tokenized_line, lowercase).split()

    @classmethod
    def _tokenize_regex(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Post-processing tokenizer for `13a` and `zh` tokenizers.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line

        """
        for _re, repl in cls._REGEX:
            line = _re.sub(repl, line)
        return ' '.join(line.split())

    @staticmethod
    def _is_chinese_char(uchar: str) -> bool:
        """Check if character is chinese.

        Args:
            uchar: input char in unicode.

        Return:
            whether the input char is a Chinese character.

        """
        return any((start <= uchar <= end for start, end in _UCODE_RANGES))

    @classmethod
    def _tokenize_base(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes an input line with the tokenizer.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line

        """
        return line

    @classmethod
    def _tokenize_13a(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes a line using a relatively minimal tokenization that is equivalent to mteval-v13a, used by WMT.

        Args:
            line: input sentence

        Return:
            tokenized sentence

        """
        line = line.replace('<skipped>', '')
        line = line.replace('-\n', '')
        line = line.replace('\n', ' ')
        if '&' in line:
            line = line.replace('&quot;', '"')
            line = line.replace('&amp;', '&')
            line = line.replace('&lt;', '<')
            line = line.replace('&gt;', '>')
        return cls._tokenize_regex(f' {line} ')

    @classmethod
    def _tokenize_zh(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenization of Chinese text.

        This is done in two steps: separate each Chinese characters (by utf-8 encoding) and afterwards tokenize the
        Chinese part (following the `13a` i.e. mteval tokenizer).
        Author: Shujian Huang huangsj@nju.edu.cn.

        Args:
            line: input sentence

        Return:
            tokenized sentence

        """
        line = line.strip()
        line_in_chars = ''
        for char in line:
            if cls._is_chinese_char(char):
                line_in_chars += ' '
                line_in_chars += char
                line_in_chars += ' '
            else:
                line_in_chars += char
        return cls._tokenize_regex(line_in_chars)

    @classmethod
    def _tokenize_international(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes a string following the official BLEU implementation.

        See github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/mteval-v14.pl#L954-L983

        In our case, the input string is expected to be just one line.
        We just tokenize on punctuation and symbols,
        except when a punctuation is preceded and followed by a digit
        (e.g. a comma/dot as a thousand/decimal separator).
        We do not recover escaped forms of punctuations such as &apos; or &gt;
        as these should never appear in MT system outputs (see issue #138)

        Note that a number (e.g., a year) followed by a dot at the end of
        sentence is NOT tokenized, i.e. the dot stays with the number because
        `s/(\\\\p{P})(\\\\P{N})/ $1 $2/g` does not match this case (unless we add a
        space after each sentence). However, this error is already in the
        original mteval-v14.pl and we want to be consistent with it.
        The error is not present in the non-international version,
        which uses `$norm_text = " $norm_text "`.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
        for _re, repl in cls._INT_REGEX:
            line = _re.sub(repl, line)
        return ' '.join(line.split())

    @classmethod
    def _tokenize_char(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes all the characters in the input line.

        Args:
            line: a segment to tokenize

        Return:
            the tokenized line

        """
        return ' '.join((char for char in line))

    @classmethod
    def _tokenize_ja_mecab(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes a Japanese string line using MeCab morphological analyzer.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
        import ipadic
        import MeCab
        tagger = MeCab.Tagger(ipadic.MECAB_ARGS + ' -Owakati')
        line = line.strip()
        return tagger.parse(line).strip()

    @classmethod
    def _tokenize_ko_mecab(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes a Korean string line using MeCab-korean morphological analyzer.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
        import mecab_ko
        import mecab_ko_dic
        tagger = mecab_ko.Tagger(mecab_ko_dic.MECAB_ARGS + ' -Owakati')
        line = line.strip()
        return tagger.parse(line).strip()

    @classmethod
    def _tokenize_flores(cls: Type['_SacreBLEUTokenizer'], line: str, tokenize: Literal['flores101', 'flores200']) -> str:
        """Tokenizes a string line using sentencepiece tokenizer.

        Args:
            line: the input string to tokenize.
            tokenize: Tokenization technique to be used.

        Return:
            The tokenized string.

        """
        import sentencepiece
        if cls.sentencepiece_processors[tokenize] is None:
            cls.sentencepiece_processors[tokenize] = sentencepiece.SentencePieceProcessor()
            file_path = os.path.join(_FLORES_LOCAL_DIR, _FLORES_MODELS_URL[tokenize].split('/')[-1])
            if not os.path.exists(file_path):
                cls.download_flores_file(tokenize)
            cls.sentencepiece_processors[tokenize].Load(file_path)
        return ' '.join(cls.sentencepiece_processors[tokenize].EncodeAsPieces(line))

    @classmethod
    def _tokenize_flores_101(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes a string line using sentencepiece tokenizer according to `FLORES-101`_ dataset.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
        return cls._tokenize_flores(line, 'flores101')

    @classmethod
    def _tokenize_flores_200(cls: Type['_SacreBLEUTokenizer'], line: str) -> str:
        """Tokenizes a string line using sentencepiece tokenizer according to `FLORES-200`_ dataset.

        Args:
            line: the input string to tokenize.

        Return:
            The tokenized string.

        """
        return cls._tokenize_flores(line, 'flores200')

    @staticmethod
    def _lower(line: str, lowercase: bool) -> str:
        if lowercase:
            return line.lower()
        return line

    @classmethod
    def _check_tokenizers_validity(cls: Type['_SacreBLEUTokenizer'], tokenize: _TokenizersLiteral) -> None:
        """Check if a supported tokenizer is chosen.

        Also check all dependencies of a given tokenizers are installed.

        """
        if tokenize not in cls._TOKENIZE_FN:
            raise ValueError(f'Unsupported tokenizer selected. Please, choose one of {list(cls._TOKENIZE_FN.keys())}')
        if tokenize == 'intl' and (not _REGEX_AVAILABLE):
            raise ModuleNotFoundError("`'intl'` tokenization requires that `regex` is installed. Use `pip install regex` or `pip install torchmetrics[text]`.")
        if tokenize == 'ja-mecab' and (not (_MECAB_AVAILABLE and _IPADIC_AVAILABLE)):
            raise ModuleNotFoundError("`'ja-mecab'` tokenization requires that `MeCab` and `ipadic` are installed. Use `pip install mecab-python3 ipadic` or `pip install torchmetrics[text]`.")
        if tokenize == 'ko-mecab' and (not (_MECAB_KO_AVAILABLE and _MECAB_KO_DIC_AVAILABLE)):
            raise ModuleNotFoundError("`'ko-mecab'` tokenization requires that `mecab_ko` and `mecab_ko_dic` are installed. Use `pip install mecab_ko mecab_ko_dic` or `pip install torchmetrics[text]`.")
        if 'flores' in tokenize and (not _SENTENCEPIECE_AVAILABLE):
            raise ModuleNotFoundError("`'flores101' and 'flores200'` tokenizations require that `sentencepiece` is installed. Use `pip install sentencepiece` or `pip install torchmetrics[text]`.")

    @staticmethod
    def download_flores_file(model_name: Literal['flores101', 'flores200']) -> None:
        """Download necessary files for `flores` tokenization via `sentencepiece`."""
        import ssl
        import urllib.request
        os.makedirs(_FLORES_LOCAL_DIR, exist_ok=True)
        model_url = _FLORES_MODELS_URL[model_name]
        file_path = os.path.join(_FLORES_LOCAL_DIR, model_url.split('/')[-1])
        try:
            with open(file_path, 'wb') as out_file, urllib.request.urlopen(model_url) as remote_file:
                out_file.write(remote_file.read())
        except ssl.SSLError as e:
            raise OSError(f'Failed to download {model_name} model.') from e