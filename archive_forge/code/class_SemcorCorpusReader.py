from nltk.corpus.reader.api import *
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from nltk.tree import Tree
class SemcorCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the SemCor Corpus.
    For access to the complete XML data structure, use the ``xml()``
    method.  For access to simple word lists and tagged word lists, use
    ``words()``, ``sents()``, ``tagged_words()``, and ``tagged_sents()``.
    """

    def __init__(self, root, fileids, wordnet, lazy=True):
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy
        self._wordnet = wordnet

    def words(self, fileids=None):
        """
        :return: the given file(s) as a list of words and punctuation symbols.
        :rtype: list(str)
        """
        return self._items(fileids, 'word', False, False, False)

    def chunks(self, fileids=None):
        """
        :return: the given file(s) as a list of chunks,
            each of which is a list of words and punctuation symbols
            that form a unit.
        :rtype: list(list(str))
        """
        return self._items(fileids, 'chunk', False, False, False)

    def tagged_chunks(self, fileids=None, tag='pos' or 'sem' or 'both'):
        """
        :return: the given file(s) as a list of tagged chunks, represented
            in tree form.
        :rtype: list(Tree)

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
        return self._items(fileids, 'chunk', False, tag != 'sem', tag != 'pos')

    def sents(self, fileids=None):
        """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of word strings.
        :rtype: list(list(str))
        """
        return self._items(fileids, 'word', True, False, False)

    def chunk_sents(self, fileids=None):
        """
        :return: the given file(s) as a list of sentences, each encoded
            as a list of chunks.
        :rtype: list(list(list(str)))
        """
        return self._items(fileids, 'chunk', True, False, False)

    def tagged_sents(self, fileids=None, tag='pos' or 'sem' or 'both'):
        """
        :return: the given file(s) as a list of sentences. Each sentence
            is represented as a list of tagged chunks (in tree form).
        :rtype: list(list(Tree))

        :param tag: `'pos'` (part of speech), `'sem'` (semantic), or `'both'`
            to indicate the kind of tags to include.  Semantic tags consist of
            WordNet lemma IDs, plus an `'NE'` node if the chunk is a named entity
            without a specific entry in WordNet.  (Named entities of type 'other'
            have no lemma.  Other chunks not in WordNet have no semantic tag.
            Punctuation tokens have `None` for their part of speech tag.)
        """
        return self._items(fileids, 'chunk', True, tag != 'sem', tag != 'pos')

    def _items(self, fileids, unit, bracket_sent, pos_tag, sem_tag):
        if unit == 'word' and (not bracket_sent):
            _ = lambda *args: LazyConcatenation((SemcorWordView if self._lazy else self._words)(*args))
        else:
            _ = SemcorWordView if self._lazy else self._words
        return concat([_(fileid, unit, bracket_sent, pos_tag, sem_tag, self._wordnet) for fileid in self.abspaths(fileids)])

    def _words(self, fileid, unit, bracket_sent, pos_tag, sem_tag):
        """
        Helper used to implement the view methods -- returns a list of
        tokens, (segmented) words, chunks, or sentences. The tokens
        and chunks may optionally be tagged (with POS and sense
        information).

        :param fileid: The name of the underlying file.
        :param unit: One of `'token'`, `'word'`, or `'chunk'`.
        :param bracket_sent: If true, include sentence bracketing.
        :param pos_tag: Whether to include part-of-speech tags.
        :param sem_tag: Whether to include semantic tags, namely WordNet lemma
            and OOV named entity status.
        """
        assert unit in ('token', 'word', 'chunk')
        result = []
        xmldoc = ElementTree.parse(fileid).getroot()
        for xmlsent in xmldoc.findall('.//s'):
            sent = []
            for xmlword in _all_xmlwords_in(xmlsent):
                itm = SemcorCorpusReader._word(xmlword, unit, pos_tag, sem_tag, self._wordnet)
                if unit == 'word':
                    sent.extend(itm)
                else:
                    sent.append(itm)
            if bracket_sent:
                result.append(SemcorSentence(xmlsent.attrib['snum'], sent))
            else:
                result.extend(sent)
        assert None not in result
        return result

    @staticmethod
    def _word(xmlword, unit, pos_tag, sem_tag, wordnet):
        tkn = xmlword.text
        if not tkn:
            tkn = ''
        lemma = xmlword.get('lemma', tkn)
        lexsn = xmlword.get('lexsn')
        if lexsn is not None:
            sense_key = lemma + '%' + lexsn
            wnpos = ('n', 'v', 'a', 'r', 's')[int(lexsn.split(':')[0]) - 1]
        else:
            sense_key = wnpos = None
        redef = xmlword.get('rdf', tkn)
        sensenum = xmlword.get('wnsn')
        isOOVEntity = 'pn' in xmlword.keys()
        pos = xmlword.get('pos')
        if unit == 'token':
            if not pos_tag and (not sem_tag):
                itm = tkn
            else:
                itm = (tkn,) + ((pos,) if pos_tag else ()) + ((lemma, wnpos, sensenum, isOOVEntity) if sem_tag else ())
            return itm
        else:
            ww = tkn.split('_')
            if unit == 'word':
                return ww
            else:
                if sensenum is not None:
                    try:
                        sense = wordnet.lemma_from_key(sense_key)
                    except Exception:
                        try:
                            sense = '%s.%s.%02d' % (lemma, wnpos, int(sensenum))
                        except ValueError:
                            sense = lemma + '.' + wnpos + '.' + sensenum
                bottom = [Tree(pos, ww)] if pos_tag else ww
                if sem_tag and isOOVEntity:
                    if sensenum is not None:
                        return Tree(sense, [Tree('NE', bottom)])
                    else:
                        return Tree('NE', bottom)
                elif sem_tag and sensenum is not None:
                    return Tree(sense, bottom)
                elif pos_tag:
                    return bottom[0]
                else:
                    return bottom