import json
import os  # required for doctests
import re
import socket
import time
from typing import List, Tuple
from nltk.internals import _java_options, config_java, find_jar_iter, java
from nltk.parse.api import ParserI
from nltk.parse.dependencygraph import DependencyGraph
from nltk.tag.api import TaggerI
from nltk.tokenize.api import TokenizerI
from nltk.tree import Tree
class CoreNLPParser(GenericCoreNLPParser):
    """
    Skip these tests if CoreNLP is likely not ready.
    >>> from nltk.test.setup_fixt import check_jar
    >>> check_jar(CoreNLPServer._JAR, env_vars=("CORENLP",), is_regex=True)

    The recommended usage of `CoreNLPParser` is using the context manager notation:
    >>> with CoreNLPServer() as server:
    ...     parser = CoreNLPParser(url=server.url)
    ...     next(
    ...         parser.raw_parse('The quick brown fox jumps over the lazy dog.')
    ...     ).pretty_print()  # doctest: +NORMALIZE_WHITESPACE
                            ROOT
                            |
                            S
            _______________|__________________________
            |                         VP               |
            |                _________|___             |
            |               |             PP           |
            |               |     ________|___         |
            NP              |    |            NP       |
        ____|__________     |    |     _______|____    |
        DT   JJ    JJ   NN  VBZ   IN   DT      JJ   NN  .
        |    |     |    |    |    |    |       |    |   |
        The quick brown fox jumps over the     lazy dog  .

    Alternatively, the server can be started using the following notation.
    Note that `CoreNLPServer` does not need to be used if the CoreNLP server is started
    outside of Python.
    >>> server = CoreNLPServer()
    >>> server.start()
    >>> parser = CoreNLPParser(url=server.url)

    >>> (parse_fox, ), (parse_wolf, ) = parser.raw_parse_sents(
    ...     [
    ...         'The quick brown fox jumps over the lazy dog.',
    ...         'The quick grey wolf jumps over the lazy fox.',
    ...     ]
    ... )

    >>> parse_fox.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
                         ROOT
                          |
                          S
           _______________|__________________________
          |                         VP               |
          |                _________|___             |
          |               |             PP           |
          |               |     ________|___         |
          NP              |    |            NP       |
      ____|__________     |    |     _______|____    |
     DT   JJ    JJ   NN  VBZ   IN   DT      JJ   NN  .
     |    |     |    |    |    |    |       |    |   |
    The quick brown fox jumps over the     lazy dog  .

    >>> parse_wolf.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
                         ROOT
                          |
                          S
           _______________|__________________________
          |                         VP               |
          |                _________|___             |
          |               |             PP           |
          |               |     ________|___         |
          NP              |    |            NP       |
      ____|_________      |    |     _______|____    |
     DT   JJ   JJ   NN   VBZ   IN   DT      JJ   NN  .
     |    |    |    |     |    |    |       |    |   |
    The quick grey wolf jumps over the     lazy fox  .

    >>> (parse_dog, ), (parse_friends, ) = parser.parse_sents(
    ...     [
    ...         "I 'm a dog".split(),
    ...         "This is my friends ' cat ( the tabby )".split(),
    ...     ]
    ... )

    >>> parse_dog.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
            ROOT
             |
             S
      _______|____
     |            VP
     |    ________|___
     NP  |            NP
     |   |         ___|___
    PRP VBP       DT      NN
     |   |        |       |
     I   'm       a      dog

    >>> parse_friends.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
         ROOT
          |
          S
      ____|___________
     |                VP
     |     ___________|_____________
     |    |                         NP
     |    |                  _______|________________________
     |    |                 NP           |        |          |
     |    |            _____|_______     |        |          |
     NP   |           NP            |    |        NP         |
     |    |     ______|_________    |    |     ___|____      |
     DT  VBZ  PRP$   NNS       POS  NN -LRB-  DT       NN  -RRB-
     |    |    |      |         |   |    |    |        |     |
    This  is   my  friends      '  cat -LRB- the     tabby -RRB-

    >>> parse_john, parse_mary, = parser.parse_text(
    ...     'John loves Mary. Mary walks.'
    ... )

    >>> parse_john.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
          ROOT
           |
           S
      _____|_____________
     |          VP       |
     |      ____|___     |
     NP    |        NP   |
     |     |        |    |
    NNP   VBZ      NNP   .
     |     |        |    |
    John loves     Mary  .

    >>> parse_mary.pretty_print()  # doctest: +NORMALIZE_WHITESPACE
          ROOT
           |
           S
      _____|____
     NP    VP   |
     |     |    |
    NNP   VBZ   .
     |     |    |
    Mary walks  .

    Special cases

    >>> next(
    ...     parser.raw_parse(
    ...         'NASIRIYA, Iraq—Iraqi doctors who treated former prisoner of war '
    ...         'Jessica Lynch have angrily dismissed claims made in her biography '
    ...         'that she was raped by her Iraqi captors.'
    ...     )
    ... ).height()
    14

    >>> next(
    ...     parser.raw_parse(
    ...         "The broader Standard & Poor's 500 Index <.SPX> was 0.46 points lower, or "
    ...         '0.05 percent, at 997.02.'
    ...     )
    ... ).height()
    11

    >>> server.stop()
    """
    _OUTPUT_FORMAT = 'penn'
    parser_annotator = 'parse'

    def make_tree(self, result):
        return Tree.fromstring(result['parse'])