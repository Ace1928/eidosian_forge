from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView

        :param fileid: The name of the underlying file.
        :param sent: If true, include sentence bracketing.
        :param tag: The name of the tagset to use, or None for no tags.
        :param strip_space: If true, strip spaces from word tokens.
        :param stem: If true, then substitute stems for words.
        