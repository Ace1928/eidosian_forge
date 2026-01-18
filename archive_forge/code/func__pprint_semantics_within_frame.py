import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _pprint_semantics_within_frame(self, vnframe, indent=''):
    """Returns a pretty printed version of semantics within frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame semantics.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
    pieces = []
    for predicate in vnframe['semantics']:
        arguments = [argument['value'] for argument in predicate['arguments']]
        pieces.append(f'{('¬' if predicate['negated'] else '')}{predicate['predicate_value']}({', '.join(arguments)})')
    return '\n'.join((f'{indent}* {piece}' for piece in pieces))