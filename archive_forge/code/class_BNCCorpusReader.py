from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader, XMLCorpusView
class BNCCorpusReader(XMLCorpusReader):
    """Corpus reader for the XML version of the British National Corpus.

    For access to the complete XML data structure, use the ``xml()``
    method.  For access to simple word lists and tagged word lists, use
    ``words()``, ``sents()``, ``tagged_words()``, and ``tagged_sents()``.

    You can obtain the full version of the BNC corpus at
    https://www.ota.ox.ac.uk/desc/2554

    If you extracted the archive to a directory called `BNC`, then you can
    instantiate the reader as::

        BNCCorpusReader(root='BNC/Texts/', fileids=r'[A-K]/\\w*/\\w*\\.xml')

    """

    def __init__(self, root, fileids, lazy=True):
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy

    def words(self, fileids=None, strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of words
            and punctuation symbols.
        :rtype: list(str)

        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        return self._views(fileids, False, None, strip_space, stem)

    def tagged_words(self, fileids=None, c5=False, strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of tagged
            words and punctuation symbols, encoded as tuples
            ``(word,tag)``.
        :rtype: list(tuple(str,str))

        :param c5: If true, then the tags used will be the more detailed
            c5 tags.  Otherwise, the simplified tags will be used.
        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        tag = 'c5' if c5 else 'pos'
        return self._views(fileids, False, tag, strip_space, stem)

    def sents(self, fileids=None, strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of
            sentences or utterances, each encoded as a list of word
            strings.
        :rtype: list(list(str))

        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        return self._views(fileids, True, None, strip_space, stem)

    def tagged_sents(self, fileids=None, c5=False, strip_space=True, stem=False):
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.
        :rtype: list(list(tuple(str,str)))

        :param c5: If true, then the tags used will be the more detailed
            c5 tags.  Otherwise, the simplified tags will be used.
        :param strip_space: If true, then strip trailing spaces from
            word tokens.  Otherwise, leave the spaces on the tokens.
        :param stem: If true, then use word stems instead of word strings.
        """
        tag = 'c5' if c5 else 'pos'
        return self._views(fileids, sent=True, tag=tag, strip_space=strip_space, stem=stem)

    def _views(self, fileids=None, sent=False, tag=False, strip_space=True, stem=False):
        """A helper function that instantiates BNCWordViews or the list of words/sentences."""
        f = BNCWordView if self._lazy else self._words
        return concat([f(fileid, sent, tag, strip_space, stem) for fileid in self.abspaths(fileids)])

    def _words(self, fileid, bracket_sent, tag, strip_space, stem):
        """
        Helper used to implement the view methods -- returns a list of
        words or a list of sentences, optionally tagged.

        :param fileid: The name of the underlying file.
        :param bracket_sent: If true, include sentence bracketing.
        :param tag: The name of the tagset to use, or None for no tags.
        :param strip_space: If true, strip spaces from word tokens.
        :param stem: If true, then substitute stems for words.
        """
        result = []
        xmldoc = ElementTree.parse(fileid).getroot()
        for xmlsent in xmldoc.findall('.//s'):
            sent = []
            for xmlword in _all_xmlwords_in(xmlsent):
                word = xmlword.text
                if not word:
                    word = ''
                if strip_space or stem:
                    word = word.strip()
                if stem:
                    word = xmlword.get('hw', word)
                if tag == 'c5':
                    word = (word, xmlword.get('c5'))
                elif tag == 'pos':
                    word = (word, xmlword.get('pos', xmlword.get('c5')))
                sent.append(word)
            if bracket_sent:
                result.append(BNCSentence(xmlsent.attrib['n'], sent))
            else:
                result.extend(sent)
        assert None not in result
        return result