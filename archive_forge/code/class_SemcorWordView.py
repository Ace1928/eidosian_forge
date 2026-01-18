from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree import Tree
class SemcorWordView(XMLCorpusView):
    """
    A stream backed corpus view specialized for use with the BNC corpus.
    """

    def __init__(self, fileid, unit, bracket_sent, pos_tag, sem_tag, wordnet):
        """
        :param fileid: The name of the underlying file.
        :param unit: One of `'token'`, `'word'`, or `'chunk'`.
        :param bracket_sent: If true, include sentence bracketing.
        :param pos_tag: Whether to include part-of-speech tags.
        :param sem_tag: Whether to include semantic tags, namely WordNet lemma
            and OOV named entity status.
        """
        if bracket_sent:
            tagspec = '.*/s'
        else:
            tagspec = '.*/s/(punc|wf)'
        self._unit = unit
        self._sent = bracket_sent
        self._pos_tag = pos_tag
        self._sem_tag = sem_tag
        self._wordnet = wordnet
        XMLCorpusView.__init__(self, fileid, tagspec)

    def handle_elt(self, elt, context):
        if self._sent:
            return self.handle_sent(elt)
        else:
            return self.handle_word(elt)

    def handle_word(self, elt):
        return SemcorCorpusReader._word(elt, self._unit, self._pos_tag, self._sem_tag, self._wordnet)

    def handle_sent(self, elt):
        sent = []
        for child in elt:
            if child.tag in ('wf', 'punc'):
                itm = self.handle_word(child)
                if self._unit == 'word':
                    sent.extend(itm)
                else:
                    sent.append(itm)
            else:
                raise ValueError('Unexpected element %s' % child.tag)
        return SemcorSentence(elt.attrib['snum'], sent)