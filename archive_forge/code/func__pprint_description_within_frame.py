import re
import textwrap
from collections import defaultdict
from nltk.corpus.reader.xmldocs import XMLCorpusReader
def _pprint_description_within_frame(self, vnframe, indent=''):
    """Returns pretty printed version of a VerbNet frame description

        Return a string containing a pretty-printed representation of
        the given VerbNet frame description.

        :param vnframe: An ElementTree containing the xml contents of
            a VerbNet frame.
        """
    description = indent + vnframe['description']['primary']
    if vnframe['description']['secondary']:
        description += ' ({})'.format(vnframe['description']['secondary'])
    return description