import re
from warnings import warn
from xml.etree import ElementTree as et
from nltk.corpus.reader import CorpusReader

        Convert a BCP-47 tag to a colon-separated string of subtag names

        >>> from nltk.corpus import bcp47
        >>> bcp47.name('ca-Latn-ES-valencia')
        'Catalan: Latin: Spain: Valencian'

        