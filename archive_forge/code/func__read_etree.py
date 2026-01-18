from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.corpus.reader.xmldocs import *
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