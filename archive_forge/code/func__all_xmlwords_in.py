from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView
def _all_xmlwords_in(elt, result=None):
    if result is None:
        result = []
    for child in elt:
        if child.tag in ('c', 'w'):
            result.append(child)
        else:
            _all_xmlwords_in(child, result)
    return result