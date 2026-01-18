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
class CoreNLPDependencyParser(GenericCoreNLPParser):
    """Dependency parser.

    Skip these tests if CoreNLP is likely not ready.
    >>> from nltk.test.setup_fixt import check_jar
    >>> check_jar(CoreNLPServer._JAR, env_vars=("CORENLP",), is_regex=True)

    The recommended usage of `CoreNLPParser` is using the context manager notation:
    >>> with CoreNLPServer() as server:
    ...     dep_parser = CoreNLPDependencyParser(url=server.url)
    ...     parse, = dep_parser.raw_parse(
    ...         'The quick brown fox jumps over the lazy dog.'
    ...     )
    ...     print(parse.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    The        DT      4       det
    quick      JJ      4       amod
    brown      JJ      4       amod
    fox        NN      5       nsubj
    jumps      VBZ     0       ROOT
    over       IN      9       case
    the        DT      9       det
    lazy       JJ      9       amod
    dog        NN      5       obl
    .  .       5       punct

    Alternatively, the server can be started using the following notation.
    Note that `CoreNLPServer` does not need to be used if the CoreNLP server is started
    outside of Python.
    >>> server = CoreNLPServer()
    >>> server.start()
    >>> dep_parser = CoreNLPDependencyParser(url=server.url)
    >>> parse, = dep_parser.raw_parse('The quick brown fox jumps over the lazy dog.')
    >>> print(parse.tree())  # doctest: +NORMALIZE_WHITESPACE
    (jumps (fox The quick brown) (dog over the lazy) .)

    >>> for governor, dep, dependent in parse.triples():
    ...     print(governor, dep, dependent)  # doctest: +NORMALIZE_WHITESPACE
    ('jumps', 'VBZ') nsubj ('fox', 'NN')
    ('fox', 'NN') det ('The', 'DT')
    ('fox', 'NN') amod ('quick', 'JJ')
    ('fox', 'NN') amod ('brown', 'JJ')
    ('jumps', 'VBZ') obl ('dog', 'NN')
    ('dog', 'NN') case ('over', 'IN')
    ('dog', 'NN') det ('the', 'DT')
    ('dog', 'NN') amod ('lazy', 'JJ')
    ('jumps', 'VBZ') punct ('.', '.')

    >>> (parse_fox, ), (parse_dog, ) = dep_parser.raw_parse_sents(
    ...     [
    ...         'The quick brown fox jumps over the lazy dog.',
    ...         'The quick grey wolf jumps over the lazy fox.',
    ...     ]
    ... )
    >>> print(parse_fox.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    The        DT      4       det
    quick      JJ      4       amod
    brown      JJ      4       amod
    fox        NN      5       nsubj
    jumps      VBZ     0       ROOT
    over       IN      9       case
    the        DT      9       det
    lazy       JJ      9       amod
    dog        NN      5       obl
    .  .       5       punct

    >>> print(parse_dog.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    The        DT      4       det
    quick      JJ      4       amod
    grey       JJ      4       amod
    wolf       NN      5       nsubj
    jumps      VBZ     0       ROOT
    over       IN      9       case
    the        DT      9       det
    lazy       JJ      9       amod
    fox        NN      5       obl
    .  .       5       punct

    >>> (parse_dog, ), (parse_friends, ) = dep_parser.parse_sents(
    ...     [
    ...         "I 'm a dog".split(),
    ...         "This is my friends ' cat ( the tabby )".split(),
    ...     ]
    ... )
    >>> print(parse_dog.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    I   PRP     4       nsubj
    'm  VBP     4       cop
    a   DT      4       det
    dog NN      0       ROOT

    >>> print(parse_friends.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    This       DT      6       nsubj
    is VBZ     6       cop
    my PRP$    4       nmod:poss
    friends    NNS     6       nmod:poss
    '  POS     4       case
    cat        NN      0       ROOT
    (  -LRB-   9       punct
    the        DT      9       det
    tabby      NN      6       dep
    )  -RRB-   9       punct

    >>> parse_john, parse_mary, = dep_parser.parse_text(
    ...     'John loves Mary. Mary walks.'
    ... )

    >>> print(parse_john.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    John       NNP     2       nsubj
    loves      VBZ     0       ROOT
    Mary       NNP     2       obj
    .  .       2       punct

    >>> print(parse_mary.to_conll(4))  # doctest: +NORMALIZE_WHITESPACE
    Mary        NNP     2       nsubj
    walks       VBZ     0       ROOT
    .   .       2       punct

    Special cases

    Non-breaking space inside of a token.

    >>> len(
    ...     next(
    ...         dep_parser.raw_parse(
    ...             'Anhalt said children typically treat a 20-ounce soda bottle as one '
    ...             'serving, while it actually contains 2 1/2 servings.'
    ...         )
    ...     ).nodes
    ... )
    23

    Phone  numbers.

    >>> len(
    ...     next(
    ...         dep_parser.raw_parse('This is not going to crash: 01 111 555.')
    ...     ).nodes
    ... )
    10

    >>> print(
    ...     next(
    ...         dep_parser.raw_parse('The underscore _ should not simply disappear.')
    ...     ).to_conll(4)
    ... )  # doctest: +NORMALIZE_WHITESPACE
    The        DT      2       det
    underscore NN      7       nsubj
    _  NFP     7       punct
    should     MD      7       aux
    not        RB      7       advmod
    simply     RB      7       advmod
    disappear  VB      0       ROOT
    .  .       7       punct

    >>> print(
    ...     next(
    ...         dep_parser.raw_parse(
    ...             'for all of its insights into the dream world of teen life , and its electronic expression through '
    ...             'cyber culture , the film gives no quarter to anyone seeking to pull a cohesive story out of its 2 '
    ...             '1/2-hour running time .'
    ...         )
    ...     ).to_conll(4)
    ... )  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    for        IN      2       case
    all        DT      24      obl
    of IN      5       case
    its        PRP$    5       nmod:poss
    insights   NNS     2       nmod
    into       IN      9       case
    the        DT      9       det
    dream      NN      9       compound
    world      NN      5       nmod
    of IN      12      case
    teen       NN      12      compound
    ...

    >>> server.stop()
    """
    _OUTPUT_FORMAT = 'conll2007'
    parser_annotator = 'depparse'

    def make_tree(self, result):
        return DependencyGraph((' '.join(n_items[1:]) for n_items in sorted(transform(result))), cell_separator=' ')