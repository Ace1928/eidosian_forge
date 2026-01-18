from __future__ import absolute_import
import sys
import re
import itertools as _itertools
from codecs import BOM_UTF8
from typing import NamedTuple, Tuple, Iterator, Iterable, List, Dict, \
from parso.python.token import PythonTokenTypes
from parso.utils import split_lines, PythonVersionInfo, parse_version_string
def _create_token_collection(version_info):
    Whitespace = '[ \\f\\t]*'
    whitespace = _compile(Whitespace)
    Comment = '#[^\\r\\n]*'
    Name = '([A-Za-z_0-9\x80-' + MAX_UNICODE + ']+)'
    Hexnumber = '0[xX](?:_?[0-9a-fA-F])+'
    Binnumber = '0[bB](?:_?[01])+'
    Octnumber = '0[oO](?:_?[0-7])+'
    Decnumber = '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    Intnumber = group(Hexnumber, Binnumber, Octnumber, Decnumber)
    Exponent = '[eE][-+]?[0-9](?:_?[0-9])*'
    Pointfloat = group('[0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?', '\\.[0-9](?:_?[0-9])*') + maybe(Exponent)
    Expfloat = '[0-9](?:_?[0-9])*' + Exponent
    Floatnumber = group(Pointfloat, Expfloat)
    Imagnumber = group('[0-9](?:_?[0-9])*[jJ]', Floatnumber + '[jJ]')
    Number = group(Imagnumber, Floatnumber, Intnumber)
    possible_prefixes = _all_string_prefixes()
    StringPrefix = group(*possible_prefixes)
    StringPrefixWithF = group(*_all_string_prefixes(include_fstring=True))
    fstring_prefixes = _all_string_prefixes(include_fstring=True, only_fstring=True)
    FStringStart = group(*fstring_prefixes)
    Single = "(?:\\\\.|[^'\\\\])*'"
    Double = '(?:\\\\.|[^"\\\\])*"'
    Single3 = "(?:\\\\.|'(?!'')|[^'\\\\])*'''"
    Double3 = '(?:\\\\.|"(?!"")|[^"\\\\])*"""'
    Triple = group(StringPrefixWithF + "'''", StringPrefixWithF + '"""')
    Operator = group('\\*\\*=?', '>>=?', '<<=?', '//=?', '->', '[+\\-*/%&@`|^!=<>]=?', '~')
    Bracket = '[][(){}]'
    special_args = ['\\.\\.\\.', '\\r\\n?', '\\n', '[;.,@]']
    if version_info >= (3, 8):
        special_args.insert(0, ':=?')
    else:
        special_args.insert(0, ':')
    Special = group(*special_args)
    Funny = group(Operator, Bracket, Special)
    ContStr = group(StringPrefix + "'[^\\r\\n'\\\\]*(?:\\\\.[^\\r\\n'\\\\]*)*" + group("'", '\\\\(?:\\r\\n?|\\n)'), StringPrefix + '"[^\\r\\n"\\\\]*(?:\\\\.[^\\r\\n"\\\\]*)*' + group('"', '\\\\(?:\\r\\n?|\\n)'))
    pseudo_extra_pool = [Comment, Triple]
    all_quotes = ('"', "'", '"""', "'''")
    if fstring_prefixes:
        pseudo_extra_pool.append(FStringStart + group(*all_quotes))
    PseudoExtras = group('\\\\(?:\\r\\n?|\\n)|\\Z', *pseudo_extra_pool)
    PseudoToken = group(Whitespace, capture=True) + group(PseudoExtras, Number, Funny, ContStr, Name, capture=True)
    endpats = {}
    for _prefix in possible_prefixes:
        endpats[_prefix + "'"] = _compile(Single)
        endpats[_prefix + '"'] = _compile(Double)
        endpats[_prefix + "'''"] = _compile(Single3)
        endpats[_prefix + '"""'] = _compile(Double3)
    single_quoted = set()
    triple_quoted = set()
    fstring_pattern_map = {}
    for t in possible_prefixes:
        for quote in ('"', "'"):
            single_quoted.add(t + quote)
        for quote in ('"""', "'''"):
            triple_quoted.add(t + quote)
    for t in fstring_prefixes:
        for quote in all_quotes:
            fstring_pattern_map[t + quote] = quote
    ALWAYS_BREAK_TOKENS = (';', 'import', 'class', 'def', 'try', 'except', 'finally', 'while', 'with', 'return', 'continue', 'break', 'del', 'pass', 'global', 'assert', 'nonlocal')
    pseudo_token_compiled = _compile(PseudoToken)
    return TokenCollection(pseudo_token_compiled, single_quoted, triple_quoted, endpats, whitespace, fstring_pattern_map, set(ALWAYS_BREAK_TOKENS))