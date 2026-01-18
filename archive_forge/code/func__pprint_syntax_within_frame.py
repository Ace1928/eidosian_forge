import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _pprint_syntax_within_frame(self, vnframe, indent=''):
    """Returns pretty printed version of syntax within a frame in a VerbNet class

        Return a string containing a pretty-printed representation of
        the given VerbNet frame syntax.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
    pieces = []
    for element in vnframe['syntax']:
        piece = element['pos_tag']
        modifier_list = []
        if 'value' in element['modifiers'] and element['modifiers']['value']:
            modifier_list.append(element['modifiers']['value'])
        modifier_list += ['{}{}'.format(restr['value'], restr['type']) for restr in element['modifiers']['selrestrs'] + element['modifiers']['synrestrs']]
        if modifier_list:
            piece += '[{}]'.format(' '.join(modifier_list))
        pieces.append(piece)
    return indent + ' '.join(pieces)