import re
from collections import defaultdict
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import ElementTree, XMLCorpusReader
from nltk.util import LazyConcatenation, LazyMap, flatten
class CHILDESCorpusReader(XMLCorpusReader):
    """
    Corpus reader for the XML version of the CHILDES corpus.
    The CHILDES corpus is available at ``https://childes.talkbank.org/``. The XML
    version of CHILDES is located at ``https://childes.talkbank.org/data-xml/``.
    Copy the needed parts of the CHILDES XML corpus into the NLTK data directory
    (``nltk_data/corpora/CHILDES/``).

    For access to the file text use the usual nltk functions,
    ``words()``, ``sents()``, ``tagged_words()`` and ``tagged_sents()``.
    """

    def __init__(self, root, fileids, lazy=True):
        XMLCorpusReader.__init__(self, root, fileids)
        self._lazy = lazy

    def words(self, fileids=None, speaker='ALL', stem=False, relation=False, strip_space=True, replace=False):
        """
        :return: the given file(s) as a list of words
        :rtype: list(str)

        :param speaker: If specified, select specific speaker(s) defined
            in the corpus. Default is 'ALL' (all participants). Common choices
            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude
            researchers)
        :param stem: If true, then use word stems instead of word strings.
        :param relation: If true, then return tuples of (stem, index,
            dependent_index)
        :param strip_space: If true, then strip trailing spaces from word
            tokens. Otherwise, leave the spaces on the tokens.
        :param replace: If true, then use the replaced (intended) word instead
            of the original word (e.g., 'wat' will be replaced with 'watch')
        """
        sent = None
        pos = False
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def tagged_words(self, fileids=None, speaker='ALL', stem=False, relation=False, strip_space=True, replace=False):
        """
        :return: the given file(s) as a list of tagged
            words and punctuation symbols, encoded as tuples
            ``(word,tag)``.
        :rtype: list(tuple(str,str))

        :param speaker: If specified, select specific speaker(s) defined
            in the corpus. Default is 'ALL' (all participants). Common choices
            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude
            researchers)
        :param stem: If true, then use word stems instead of word strings.
        :param relation: If true, then return tuples of (stem, index,
            dependent_index)
        :param strip_space: If true, then strip trailing spaces from word
            tokens. Otherwise, leave the spaces on the tokens.
        :param replace: If true, then use the replaced (intended) word instead
            of the original word (e.g., 'wat' will be replaced with 'watch')
        """
        sent = None
        pos = True
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def sents(self, fileids=None, speaker='ALL', stem=False, relation=None, strip_space=True, replace=False):
        """
        :return: the given file(s) as a list of sentences or utterances, each
            encoded as a list of word strings.
        :rtype: list(list(str))

        :param speaker: If specified, select specific speaker(s) defined
            in the corpus. Default is 'ALL' (all participants). Common choices
            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude
            researchers)
        :param stem: If true, then use word stems instead of word strings.
        :param relation: If true, then return tuples of ``(str,pos,relation_list)``.
            If there is manually-annotated relation info, it will return
            tuples of ``(str,pos,test_relation_list,str,pos,gold_relation_list)``
        :param strip_space: If true, then strip trailing spaces from word
            tokens. Otherwise, leave the spaces on the tokens.
        :param replace: If true, then use the replaced (intended) word instead
            of the original word (e.g., 'wat' will be replaced with 'watch')
        """
        sent = True
        pos = False
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def tagged_sents(self, fileids=None, speaker='ALL', stem=False, relation=None, strip_space=True, replace=False):
        """
        :return: the given file(s) as a list of
            sentences, each encoded as a list of ``(word,tag)`` tuples.
        :rtype: list(list(tuple(str,str)))

        :param speaker: If specified, select specific speaker(s) defined
            in the corpus. Default is 'ALL' (all participants). Common choices
            are 'CHI' (the child), 'MOT' (mother), ['CHI','MOT'] (exclude
            researchers)
        :param stem: If true, then use word stems instead of word strings.
        :param relation: If true, then return tuples of ``(str,pos,relation_list)``.
            If there is manually-annotated relation info, it will return
            tuples of ``(str,pos,test_relation_list,str,pos,gold_relation_list)``
        :param strip_space: If true, then strip trailing spaces from word
            tokens. Otherwise, leave the spaces on the tokens.
        :param replace: If true, then use the replaced (intended) word instead
            of the original word (e.g., 'wat' will be replaced with 'watch')
        """
        sent = True
        pos = True
        if not self._lazy:
            return [self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace) for fileid in self.abspaths(fileids)]
        get_words = lambda fileid: self._get_words(fileid, speaker, sent, stem, relation, pos, strip_space, replace)
        return LazyConcatenation(LazyMap(get_words, self.abspaths(fileids)))

    def corpus(self, fileids=None):
        """
        :return: the given file(s) as a dict of ``(corpus_property_key, value)``
        :rtype: list(dict)
        """
        if not self._lazy:
            return [self._get_corpus(fileid) for fileid in self.abspaths(fileids)]
        return LazyMap(self._get_corpus, self.abspaths(fileids))

    def _get_corpus(self, fileid):
        results = dict()
        xmldoc = ElementTree.parse(fileid).getroot()
        for key, value in xmldoc.items():
            results[key] = value
        return results

    def participants(self, fileids=None):
        """
        :return: the given file(s) as a dict of
            ``(participant_property_key, value)``
        :rtype: list(dict)
        """
        if not self._lazy:
            return [self._get_participants(fileid) for fileid in self.abspaths(fileids)]
        return LazyMap(self._get_participants, self.abspaths(fileids))

    def _get_participants(self, fileid):

        def dictOfDicts():
            return defaultdict(dictOfDicts)
        xmldoc = ElementTree.parse(fileid).getroot()
        pat = dictOfDicts()
        for participant in xmldoc.findall(f'.//{{{NS}}}Participants/{{{NS}}}participant'):
            for key, value in participant.items():
                pat[participant.get('id')][key] = value
        return pat

    def age(self, fileids=None, speaker='CHI', month=False):
        """
        :return: the given file(s) as string or int
        :rtype: list or int

        :param month: If true, return months instead of year-month-date
        """
        if not self._lazy:
            return [self._get_age(fileid, speaker, month) for fileid in self.abspaths(fileids)]
        get_age = lambda fileid: self._get_age(fileid, speaker, month)
        return LazyMap(get_age, self.abspaths(fileids))

    def _get_age(self, fileid, speaker, month):
        xmldoc = ElementTree.parse(fileid).getroot()
        for pat in xmldoc.findall(f'.//{{{NS}}}Participants/{{{NS}}}participant'):
            try:
                if pat.get('id') == speaker:
                    age = pat.get('age')
                    if month:
                        age = self.convert_age(age)
                    return age
            except (TypeError, AttributeError) as e:
                return None

    def convert_age(self, age_year):
        """Caclculate age in months from a string in CHILDES format"""
        m = re.match('P(\\d+)Y(\\d+)M?(\\d?\\d?)D?', age_year)
        age_month = int(m.group(1)) * 12 + int(m.group(2))
        try:
            if int(m.group(3)) > 15:
                age_month += 1
        except ValueError as e:
            pass
        return age_month

    def MLU(self, fileids=None, speaker='CHI'):
        """
        :return: the given file(s) as a floating number
        :rtype: list(float)
        """
        if not self._lazy:
            return [self._getMLU(fileid, speaker=speaker) for fileid in self.abspaths(fileids)]
        get_MLU = lambda fileid: self._getMLU(fileid, speaker=speaker)
        return LazyMap(get_MLU, self.abspaths(fileids))

    def _getMLU(self, fileid, speaker):
        sents = self._get_words(fileid, speaker=speaker, sent=True, stem=True, relation=False, pos=True, strip_space=True, replace=True)
        results = []
        lastSent = []
        numFillers = 0
        sentDiscount = 0
        for sent in sents:
            posList = [pos for word, pos in sent]
            if any((pos == 'unk' for pos in posList)):
                continue
            elif sent == []:
                continue
            elif sent == lastSent:
                continue
            else:
                results.append([word for word, pos in sent])
                if len({'co', None}.intersection(posList)) > 0:
                    numFillers += posList.count('co')
                    numFillers += posList.count(None)
                    sentDiscount += 1
            lastSent = sent
        try:
            thisWordList = flatten(results)
            numWords = len(flatten([word.split('-') for word in thisWordList])) - numFillers
            numSents = len(results) - sentDiscount
            mlu = numWords / numSents
        except ZeroDivisionError:
            mlu = 0
        return mlu

    def _get_words(self, fileid, speaker, sent, stem, relation, pos, strip_space, replace):
        if isinstance(speaker, str) and speaker != 'ALL':
            speaker = [speaker]
        xmldoc = ElementTree.parse(fileid).getroot()
        results = []
        for xmlsent in xmldoc.findall('.//{%s}u' % NS):
            sents = []
            if speaker == 'ALL' or xmlsent.get('who') in speaker:
                for xmlword in xmlsent.findall('.//{%s}w' % NS):
                    infl = None
                    suffixStem = None
                    suffixTag = None
                    if replace and xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}replacement'):
                        xmlword = xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}replacement/{{{NS}}}w')
                    elif replace and xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}wk'):
                        xmlword = xmlsent.find(f'.//{{{NS}}}w/{{{NS}}}wk')
                    if xmlword.text:
                        word = xmlword.text
                    else:
                        word = ''
                    if strip_space:
                        word = word.strip()
                    if relation or stem:
                        try:
                            xmlstem = xmlword.find('.//{%s}stem' % NS)
                            word = xmlstem.text
                        except AttributeError as e:
                            pass
                        try:
                            xmlinfl = xmlword.find(f'.//{{{NS}}}mor/{{{NS}}}mw/{{{NS}}}mk')
                            word += '-' + xmlinfl.text
                        except:
                            pass
                        try:
                            xmlsuffix = xmlword.find('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}stem' % (NS, NS, NS, NS))
                            suffixStem = xmlsuffix.text
                        except AttributeError:
                            suffixStem = ''
                        if suffixStem:
                            word += '~' + suffixStem
                    if relation or pos:
                        try:
                            xmlpos = xmlword.findall('.//{%s}c' % NS)
                            xmlpos2 = xmlword.findall('.//{%s}s' % NS)
                            if xmlpos2 != []:
                                tag = xmlpos[0].text + ':' + xmlpos2[0].text
                            else:
                                tag = xmlpos[0].text
                        except (AttributeError, IndexError) as e:
                            tag = ''
                        try:
                            xmlsuffixpos = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}c' % (NS, NS, NS, NS, NS))
                            xmlsuffixpos2 = xmlword.findall('.//{%s}mor/{%s}mor-post/{%s}mw/{%s}pos/{%s}s' % (NS, NS, NS, NS, NS))
                            if xmlsuffixpos2:
                                suffixTag = xmlsuffixpos[0].text + ':' + xmlsuffixpos2[0].text
                            else:
                                suffixTag = xmlsuffixpos[0].text
                        except:
                            pass
                        if suffixTag:
                            tag += '~' + suffixTag
                        word = (word, tag)
                    if relation == True:
                        for xmlstem_rel in xmlword.findall(f'.//{{{NS}}}mor/{{{NS}}}gra'):
                            if not xmlstem_rel.get('type') == 'grt':
                                word = (word[0], word[1], xmlstem_rel.get('index') + '|' + xmlstem_rel.get('head') + '|' + xmlstem_rel.get('relation'))
                            else:
                                word = (word[0], word[1], word[2], word[0], word[1], xmlstem_rel.get('index') + '|' + xmlstem_rel.get('head') + '|' + xmlstem_rel.get('relation'))
                        try:
                            for xmlpost_rel in xmlword.findall(f'.//{{{NS}}}mor/{{{NS}}}mor-post/{{{NS}}}gra'):
                                if not xmlpost_rel.get('type') == 'grt':
                                    suffixStem = (suffixStem[0], suffixStem[1], xmlpost_rel.get('index') + '|' + xmlpost_rel.get('head') + '|' + xmlpost_rel.get('relation'))
                                else:
                                    suffixStem = (suffixStem[0], suffixStem[1], suffixStem[2], suffixStem[0], suffixStem[1], xmlpost_rel.get('index') + '|' + xmlpost_rel.get('head') + '|' + xmlpost_rel.get('relation'))
                        except:
                            pass
                    sents.append(word)
                if sent or relation:
                    results.append(sents)
                else:
                    results.extend(sents)
        return LazyMap(lambda x: x, results)
    "\n    The base URL for viewing files on the childes website. This\n    shouldn't need to be changed, unless CHILDES changes the configuration\n    of their server or unless the user sets up their own corpus webserver.\n    "
    childes_url_base = 'https://childes.talkbank.org/browser/index.php?url='

    def webview_file(self, fileid, urlbase=None):
        """Map a corpus file to its web version on the CHILDES website,
        and open it in a web browser.

        The complete URL to be used is:
            childes.childes_url_base + urlbase + fileid.replace('.xml', '.cha')

        If no urlbase is passed, we try to calculate it.  This
        requires that the childes corpus was set up to mirror the
        folder hierarchy under childes.psy.cmu.edu/data-xml/, e.g.:
        nltk_data/corpora/childes/Eng-USA/Cornell/??? or
        nltk_data/corpora/childes/Romance/Spanish/Aguirre/???

        The function first looks (as a special case) if "Eng-USA" is
        on the path consisting of <corpus root>+fileid; then if
        "childes", possibly followed by "data-xml", appears. If neither
        one is found, we use the unmodified fileid and hope for the best.
        If this is not right, specify urlbase explicitly, e.g., if the
        corpus root points to the Cornell folder, urlbase='Eng-USA/Cornell'.
        """
        import webbrowser
        if urlbase:
            path = urlbase + '/' + fileid
        else:
            full = self.root + '/' + fileid
            full = re.sub('\\\\', '/', full)
            if '/childes/' in full.lower():
                path = re.findall('(?i)/childes(?:/data-xml)?/(.*)\\.xml', full)[0]
            elif 'eng-usa' in full.lower():
                path = 'Eng-USA/' + re.findall('/(?i)Eng-USA/(.*)\\.xml', full)[0]
            else:
                path = fileid
        if path.endswith('.xml'):
            path = path[:-4]
        if not path.endswith('.cha'):
            path = path + '.cha'
        url = self.childes_url_base + path
        webbrowser.open_new_tab(url)
        print('Opening in browser:', url)