from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
class RTECorpusReader(XMLCorpusReader):
    """
    Corpus reader for corpora in RTE challenges.

    This is just a wrapper around the XMLCorpusReader. See module docstring above for the expected
    structure of input documents.
    """

    def _read_etree(self, doc):
        """
        Map the XML input into an RTEPair.

        This uses the ``getiterator()`` method from the ElementTree package to
        find all the ``<pair>`` elements.

        :param doc: a parsed XML document
        :rtype: list(RTEPair)
        """
        try:
            challenge = doc.attrib['challenge']
        except KeyError:
            challenge = None
        pairiter = doc.iter('pair')
        return [RTEPair(pair, challenge=challenge) for pair in pairiter]

    def pairs(self, fileids):
        """
        Build a list of RTEPairs from a RTE corpus.

        :param fileids: a list of RTE corpus fileids
        :type: list
        :rtype: list(RTEPair)
        """
        if isinstance(fileids, str):
            fileids = [fileids]
        return concat([self._read_etree(self.xml(fileid)) for fileid in fileids])