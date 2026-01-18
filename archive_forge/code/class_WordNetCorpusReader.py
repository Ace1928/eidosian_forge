import math
import os
import re
import warnings
from collections import defaultdict, deque
from functools import total_ordering
from itertools import chain, islice
from operator import itemgetter
from nltk.corpus.reader import CorpusReader
from nltk.internals import deprecated
from nltk.probability import FreqDist
from nltk.util import binary_search_file as _binary_search_file
class WordNetCorpusReader(CorpusReader):
    """
    A corpus reader used to access wordnet or its variants.
    """
    _ENCODING = 'utf8'
    ADJ, ADJ_SAT, ADV, NOUN, VERB = ('a', 's', 'r', 'n', 'v')
    _FILEMAP = {ADJ: 'adj', ADV: 'adv', NOUN: 'noun', VERB: 'verb'}
    _pos_numbers = {NOUN: 1, VERB: 2, ADJ: 3, ADV: 4, ADJ_SAT: 5}
    _pos_names = dict((tup[::-1] for tup in _pos_numbers.items()))
    _FILES = ('cntlist.rev', 'lexnames', 'index.sense', 'index.adj', 'index.adv', 'index.noun', 'index.verb', 'data.adj', 'data.adv', 'data.noun', 'data.verb', 'adj.exc', 'adv.exc', 'noun.exc', 'verb.exc')

    def __init__(self, root, omw_reader):
        """
        Construct a new wordnet corpus reader, with the given root
        directory.
        """
        super().__init__(root, self._FILES, encoding=self._ENCODING)
        self._lemma_pos_offset_map = defaultdict(dict)
        self._synset_offset_cache = defaultdict(dict)
        self._max_depth = defaultdict(dict)
        self._omw_reader = omw_reader
        self._exomw_reader = None
        self.provenances = defaultdict(str)
        self.provenances['eng'] = ''
        if self._omw_reader is None:
            warnings.warn('The multilingual functions are not available with this Wordnet version')
        self.omw_langs = set()
        self._lang_data = defaultdict(list)
        self._data_file_map = {}
        self._exception_map = {}
        self._lexnames = []
        self._key_count_file = None
        self._key_synset_file = None
        with self.open('lexnames') as fp:
            for i, line in enumerate(fp):
                index, lexname, _ = line.split()
                assert int(index) == i
                self._lexnames.append(lexname)
        self._load_lemma_pos_offset_map()
        self._load_exception_map()
        self.nomap = []
        self.splits = {}
        self.map30 = self.map_wn30()
        self.lg_attrs = ['lemma', 'none', 'def', 'exe']

    def index_sense(self, version=None):
        """Read sense key to synset id mapping from index.sense file in corpus directory"""
        fn = 'index.sense'
        if version:
            from nltk.corpus import CorpusReader, LazyCorpusLoader
            ixreader = LazyCorpusLoader(version, CorpusReader, '.*/' + fn)
        else:
            ixreader = self
        with ixreader.open(fn) as fp:
            sensekey_map = {}
            for line in fp:
                fields = line.strip().split()
                sensekey = fields[0]
                pos = self._pos_names[int(sensekey.split('%')[1].split(':')[0])]
                sensekey_map[sensekey] = f'{fields[1]}-{pos}'
        return sensekey_map

    def map_to_many(self):
        sensekey_map1 = self.index_sense('wordnet')
        sensekey_map2 = self.index_sense()
        synset_to_many = {}
        for synsetid in set(sensekey_map1.values()):
            synset_to_many[synsetid] = []
        for sensekey in set(sensekey_map1.keys()).intersection(set(sensekey_map2.keys())):
            source = sensekey_map1[sensekey]
            target = sensekey_map2[sensekey]
            synset_to_many[source].append(target)
        return synset_to_many

    def map_to_one(self):
        synset_to_many = self.map_to_many()
        synset_to_one = {}
        for source in synset_to_many:
            candidates_bag = synset_to_many[source]
            if candidates_bag:
                candidates_set = set(candidates_bag)
                if len(candidates_set) == 1:
                    target = candidates_bag[0]
                else:
                    counts = []
                    for candidate in candidates_set:
                        counts.append((candidates_bag.count(candidate), candidate))
                    self.splits[source] = counts
                    target = max(counts)[1]
                synset_to_one[source] = target
                if source[-1] == 's':
                    synset_to_one[f'{source[:-1]}a'] = target
            else:
                self.nomap.append(source)
        return synset_to_one

    def map_wn30(self):
        """Mapping from Wordnet 3.0 to currently loaded Wordnet version"""
        if self.get_version() == '3.0':
            return None
        else:
            return self.map_to_one()

    def of2ss(self, of):
        """take an id and return the synsets"""
        return self.synset_from_pos_and_offset(of[-1], int(of[:8]))

    def ss2of(self, ss):
        """return the ID of the synset"""
        if ss:
            return f'{ss.offset():08d}-{ss.pos()}'

    def _load_lang_data(self, lang):
        """load the wordnet data of the requested language from the file to
        the cache, _lang_data"""
        if lang in self._lang_data:
            return
        if self._omw_reader and (not self.omw_langs):
            self.add_omw()
        if lang not in self.langs():
            raise WordNetError('Language is not supported.')
        if self._exomw_reader and lang not in self.omw_langs:
            reader = self._exomw_reader
        else:
            reader = self._omw_reader
        prov = self.provenances[lang]
        if prov in ['cldr', 'wikt']:
            prov2 = prov
        else:
            prov2 = 'data'
        with reader.open(f'{prov}/wn-{prov2}-{lang.split('_')[0]}.tab') as fp:
            self.custom_lemmas(fp, lang)
        self.disable_custom_lemmas(lang)

    def add_provs(self, reader):
        """Add languages from Multilingual Wordnet to the provenance dictionary"""
        fileids = reader.fileids()
        for fileid in fileids:
            prov, langfile = os.path.split(fileid)
            file_name, file_extension = os.path.splitext(langfile)
            if file_extension == '.tab':
                lang = file_name.split('-')[-1]
                if lang in self.provenances or prov in ['cldr', 'wikt']:
                    lang = f'{lang}_{prov}'
                self.provenances[lang] = prov

    def add_omw(self):
        self.add_provs(self._omw_reader)
        self.omw_langs = set(self.provenances.keys())

    def add_exomw(self):
        """
        Add languages from Extended OMW

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> wn.add_exomw()
        >>> print(wn.synset('intrinsically.r.01').lemmas(lang="eng_wikt"))
        [Lemma('intrinsically.r.01.per_se'), Lemma('intrinsically.r.01.as_such')]
        """
        from nltk.corpus import extended_omw
        self.add_omw()
        self._exomw_reader = extended_omw
        self.add_provs(self._exomw_reader)

    def langs(self):
        """return a list of languages supported by Multilingual Wordnet"""
        return list(self.provenances.keys())

    def _load_lemma_pos_offset_map(self):
        for suffix in self._FILEMAP.values():
            with self.open('index.%s' % suffix) as fp:
                for i, line in enumerate(fp):
                    if line.startswith(' '):
                        continue
                    _iter = iter(line.split())

                    def _next_token():
                        return next(_iter)
                    try:
                        lemma = _next_token()
                        pos = _next_token()
                        n_synsets = int(_next_token())
                        assert n_synsets > 0
                        n_pointers = int(_next_token())
                        [_next_token() for _ in range(n_pointers)]
                        n_senses = int(_next_token())
                        assert n_synsets == n_senses
                        _next_token()
                        synset_offsets = [int(_next_token()) for _ in range(n_synsets)]
                    except (AssertionError, ValueError) as e:
                        tup = ('index.%s' % suffix, i + 1, e)
                        raise WordNetError('file %s, line %i: %s' % tup) from e
                    self._lemma_pos_offset_map[lemma][pos] = synset_offsets
                    if pos == ADJ:
                        self._lemma_pos_offset_map[lemma][ADJ_SAT] = synset_offsets

    def _load_exception_map(self):
        for pos, suffix in self._FILEMAP.items():
            self._exception_map[pos] = {}
            with self.open('%s.exc' % suffix) as fp:
                for line in fp:
                    terms = line.split()
                    self._exception_map[pos][terms[0]] = terms[1:]
        self._exception_map[ADJ_SAT] = self._exception_map[ADJ]

    def _compute_max_depth(self, pos, simulate_root):
        """
        Compute the max depth for the given part of speech.  This is
        used by the lch similarity metric.
        """
        depth = 0
        for ii in self.all_synsets(pos):
            try:
                depth = max(depth, ii.max_depth())
            except RuntimeError:
                print(ii)
        if simulate_root:
            depth += 1
        self._max_depth[pos] = depth

    def get_version(self):
        fh = self._data_file(ADJ)
        fh.seek(0)
        for line in fh:
            match = re.search('Word[nN]et (\\d+|\\d+\\.\\d+) Copyright', line)
            if match is not None:
                version = match.group(1)
                fh.seek(0)
                return version

    def lemma(self, name, lang='eng'):
        """Return lemma object that matches the name"""
        separator = SENSENUM_RE.search(name).end()
        synset_name, lemma_name = (name[:separator - 1], name[separator:])
        synset = self.synset(synset_name)
        for lemma in synset.lemmas(lang):
            if lemma._name == lemma_name:
                return lemma
        raise WordNetError(f'No lemma {lemma_name!r} in {synset_name!r}')

    def lemma_from_key(self, key):
        key = key.lower()
        lemma_name, lex_sense = key.split('%')
        pos_number, lexname_index, lex_id, _, _ = lex_sense.split(':')
        pos = self._pos_names[int(pos_number)]
        if self._key_synset_file is None:
            self._key_synset_file = self.open('index.sense')
        synset_line = _binary_search_file(self._key_synset_file, key)
        if not synset_line:
            raise WordNetError('No synset found for key %r' % key)
        offset = int(synset_line.split()[1])
        synset = self.synset_from_pos_and_offset(pos, offset)
        for lemma in synset._lemmas:
            if lemma._key == key:
                return lemma
        raise WordNetError('No lemma found for for key %r' % key)

    def synset(self, name):
        lemma, pos, synset_index_str = name.lower().rsplit('.', 2)
        synset_index = int(synset_index_str) - 1
        try:
            offset = self._lemma_pos_offset_map[lemma][pos][synset_index]
        except KeyError as e:
            raise WordNetError(f'No lemma {lemma!r} with part of speech {pos!r}') from e
        except IndexError as e:
            n_senses = len(self._lemma_pos_offset_map[lemma][pos])
            raise WordNetError(f'Lemma {lemma!r} with part of speech {pos!r} only has {n_senses} {('sense' if n_senses == 1 else 'senses')}') from e
        synset = self.synset_from_pos_and_offset(pos, offset)
        if pos == 's' and synset._pos == 'a':
            message = 'Adjective satellite requested but only plain adjective found for lemma %r'
            raise WordNetError(message % lemma)
        assert synset._pos == pos or (pos == 'a' and synset._pos == 's')
        return synset

    def _data_file(self, pos):
        """
        Return an open file pointer for the data file for the given
        part of speech.
        """
        if pos == ADJ_SAT:
            pos = ADJ
        if self._data_file_map.get(pos) is None:
            fileid = 'data.%s' % self._FILEMAP[pos]
            self._data_file_map[pos] = self.open(fileid)
        return self._data_file_map[pos]

    def synset_from_pos_and_offset(self, pos, offset):
        """
        - pos: The synset's part of speech, matching one of the module level
          attributes ADJ, ADJ_SAT, ADV, NOUN or VERB ('a', 's', 'r', 'n', or 'v').
        - offset: The byte offset of this synset in the WordNet dict file
          for this pos.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.synset_from_pos_and_offset('n', 1740))
        Synset('entity.n.01')
        """
        if offset in self._synset_offset_cache[pos]:
            return self._synset_offset_cache[pos][offset]
        data_file = self._data_file(pos)
        data_file.seek(offset)
        data_file_line = data_file.readline()
        line_offset = data_file_line[:8]
        if line_offset.isalnum() and line_offset == f'{'0' * (8 - len(str(offset)))}{str(offset)}':
            synset = self._synset_from_pos_and_line(pos, data_file_line)
            assert synset._offset == offset
            self._synset_offset_cache[pos][offset] = synset
        else:
            synset = None
            warnings.warn(f'No WordNet synset found for pos={pos} at offset={offset}.')
        data_file.seek(0)
        return synset

    @deprecated('Use public method synset_from_pos_and_offset() instead')
    def _synset_from_pos_and_offset(self, *args, **kwargs):
        """
        Hack to help people like the readers of
        https://stackoverflow.com/a/27145655/1709587
        who were using this function before it was officially a public method
        """
        return self.synset_from_pos_and_offset(*args, **kwargs)

    def _synset_from_pos_and_line(self, pos, data_file_line):
        synset = Synset(self)
        try:
            columns_str, gloss = data_file_line.strip().split('|')
            definition = re.sub('[\\"].*?[\\"]', '', gloss).strip()
            examples = re.findall('"([^"]*)"', gloss)
            for example in examples:
                synset._examples.append(example)
            synset._definition = definition.strip('; ')
            _iter = iter(columns_str.split())

            def _next_token():
                return next(_iter)
            synset._offset = int(_next_token())
            lexname_index = int(_next_token())
            synset._lexname = self._lexnames[lexname_index]
            synset._pos = _next_token()
            n_lemmas = int(_next_token(), 16)
            for _ in range(n_lemmas):
                lemma_name = _next_token()
                lex_id = int(_next_token(), 16)
                m = re.match('(.*?)(\\(.*\\))?$', lemma_name)
                lemma_name, syn_mark = m.groups()
                lemma = Lemma(self, synset, lemma_name, lexname_index, lex_id, syn_mark)
                synset._lemmas.append(lemma)
                synset._lemma_names.append(lemma._name)
            n_pointers = int(_next_token())
            for _ in range(n_pointers):
                symbol = _next_token()
                offset = int(_next_token())
                pos = _next_token()
                lemma_ids_str = _next_token()
                if lemma_ids_str == '0000':
                    synset._pointers[symbol].add((pos, offset))
                else:
                    source_index = int(lemma_ids_str[:2], 16) - 1
                    target_index = int(lemma_ids_str[2:], 16) - 1
                    source_lemma_name = synset._lemmas[source_index]._name
                    lemma_pointers = synset._lemma_pointers
                    tups = lemma_pointers[source_lemma_name, symbol]
                    tups.append((pos, offset, target_index))
            try:
                frame_count = int(_next_token())
            except StopIteration:
                pass
            else:
                for _ in range(frame_count):
                    plus = _next_token()
                    assert plus == '+'
                    frame_number = int(_next_token())
                    frame_string_fmt = VERB_FRAME_STRINGS[frame_number]
                    lemma_number = int(_next_token(), 16)
                    if lemma_number == 0:
                        synset._frame_ids.append(frame_number)
                        for lemma in synset._lemmas:
                            lemma._frame_ids.append(frame_number)
                            lemma._frame_strings.append(frame_string_fmt % lemma._name)
                    else:
                        lemma = synset._lemmas[lemma_number - 1]
                        lemma._frame_ids.append(frame_number)
                        lemma._frame_strings.append(frame_string_fmt % lemma._name)
        except ValueError as e:
            raise WordNetError(f'line {data_file_line!r}: {e}') from e
        for lemma in synset._lemmas:
            if synset._pos == ADJ_SAT:
                head_lemma = synset.similar_tos()[0]._lemmas[0]
                head_name = head_lemma._name
                head_id = '%02d' % head_lemma._lex_id
            else:
                head_name = head_id = ''
            tup = (lemma._name, WordNetCorpusReader._pos_numbers[synset._pos], lemma._lexname_index, lemma._lex_id, head_name, head_id)
            lemma._key = ('%s%%%d:%02d:%02d:%s:%s' % tup).lower()
        lemma_name = synset._lemmas[0]._name.lower()
        offsets = self._lemma_pos_offset_map[lemma_name][synset._pos]
        sense_index = offsets.index(synset._offset)
        tup = (lemma_name, synset._pos, sense_index + 1)
        synset._name = '%s.%s.%02i' % tup
        return synset

    def synset_from_sense_key(self, sense_key):
        """
        Retrieves synset based on a given sense_key. Sense keys can be
        obtained from lemma.key()

        From https://wordnet.princeton.edu/documentation/senseidx5wn:
        A sense_key is represented as::

            lemma % lex_sense (e.g. 'dog%1:18:01::')

        where lex_sense is encoded as::

            ss_type:lex_filenum:lex_id:head_word:head_id

        :lemma:       ASCII text of word/collocation, in lower case
        :ss_type:     synset type for the sense (1 digit int)
                      The synset type is encoded as follows::

                          1    NOUN
                          2    VERB
                          3    ADJECTIVE
                          4    ADVERB
                          5    ADJECTIVE SATELLITE
        :lex_filenum: name of lexicographer file containing the synset for the sense (2 digit int)
        :lex_id:      when paired with lemma, uniquely identifies a sense in the lexicographer file (2 digit int)
        :head_word:   lemma of the first word in satellite's head synset
                      Only used if sense is in an adjective satellite synset
        :head_id:     uniquely identifies sense in a lexicographer file when paired with head_word
                      Only used if head_word is present (2 digit int)

        >>> import nltk
        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.synset_from_sense_key("drive%1:04:03::"))
        Synset('drive.n.06')

        >>> print(wn.synset_from_sense_key("driving%1:04:03::"))
        Synset('drive.n.06')
        """
        return self.lemma_from_key(sense_key).synset()

    def synsets(self, lemma, pos=None, lang='eng', check_exceptions=True):
        """Load all synsets with a given lemma and part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        If lang is specified, all the synsets associated with the lemma name
        of that language will be returned.
        """
        lemma = lemma.lower()
        if lang == 'eng':
            get_synset = self.synset_from_pos_and_offset
            index = self._lemma_pos_offset_map
            if pos is None:
                pos = POS_LIST
            return [get_synset(p, offset) for p in pos for form in self._morphy(lemma, p, check_exceptions) for offset in index[form].get(p, [])]
        else:
            self._load_lang_data(lang)
            synset_list = []
            if lemma in self._lang_data[lang][1]:
                for l in self._lang_data[lang][1][lemma]:
                    if pos is not None and l[-1] != pos:
                        continue
                    synset_list.append(self.of2ss(l))
            return synset_list

    def lemmas(self, lemma, pos=None, lang='eng'):
        """Return all Lemma objects with a name matching the specified lemma
        name and part of speech tag. Matches any part of speech tag if none is
        specified."""
        lemma = lemma.lower()
        if lang == 'eng':
            return [lemma_obj for synset in self.synsets(lemma, pos) for lemma_obj in synset.lemmas() if lemma_obj.name().lower() == lemma]
        else:
            self._load_lang_data(lang)
            lemmas = []
            syn = self.synsets(lemma, lang=lang)
            for s in syn:
                if pos is not None and s.pos() != pos:
                    continue
                for lemma_obj in s.lemmas(lang=lang):
                    if lemma_obj.name().lower() == lemma:
                        lemmas.append(lemma_obj)
            return lemmas

    def all_lemma_names(self, pos=None, lang='eng'):
        """Return all lemma names for all synsets for the given
        part of speech tag and language or languages. If pos is
        not specified, all synsets for all parts of speech will
        be used."""
        if lang == 'eng':
            if pos is None:
                return iter(self._lemma_pos_offset_map)
            else:
                return (lemma for lemma in self._lemma_pos_offset_map if pos in self._lemma_pos_offset_map[lemma])
        else:
            self._load_lang_data(lang)
            lemma = []
            for i in self._lang_data[lang][0]:
                if pos is not None and i[-1] != pos:
                    continue
                lemma.extend(self._lang_data[lang][0][i])
            lemma = iter(set(lemma))
            return lemma

    def all_omw_synsets(self, pos=None, lang=None):
        if lang not in self.langs():
            return None
        self._load_lang_data(lang)
        for of in self._lang_data[lang][0]:
            if not pos or of[-1] == pos:
                ss = self.of2ss(of)
                if ss:
                    yield ss

    def all_synsets(self, pos=None, lang='eng'):
        """Iterate over all synsets with a given part of speech tag.
        If no pos is specified, all synsets for all parts of speech
        will be loaded.
        """
        if lang == 'eng':
            return self.all_eng_synsets(pos=pos)
        else:
            return self.all_omw_synsets(pos=pos, lang=lang)

    def all_eng_synsets(self, pos=None):
        if pos is None:
            pos_tags = self._FILEMAP.keys()
        else:
            pos_tags = [pos]
        cache = self._synset_offset_cache
        from_pos_and_line = self._synset_from_pos_and_line
        for pos_tag in pos_tags:
            if pos_tag == ADJ_SAT:
                pos_file = ADJ
            else:
                pos_file = pos_tag
            fileid = 'data.%s' % self._FILEMAP[pos_file]
            data_file = self.open(fileid)
            try:
                offset = data_file.tell()
                line = data_file.readline()
                while line:
                    if not line[0].isspace():
                        if offset in cache[pos_tag]:
                            synset = cache[pos_tag][offset]
                        else:
                            synset = from_pos_and_line(pos_tag, line)
                            cache[pos_tag][offset] = synset
                        if pos_tag == ADJ_SAT and synset._pos == ADJ_SAT:
                            yield synset
                        elif pos_tag != ADJ_SAT:
                            yield synset
                    offset = data_file.tell()
                    line = data_file.readline()
            except:
                data_file.close()
                raise
            else:
                data_file.close()

    def words(self, lang='eng'):
        """return lemmas of the given language as list of words"""
        return self.all_lemma_names(lang=lang)

    def synonyms(self, word, lang='eng'):
        """return nested list with the synonyms of the different senses of word in the given language"""
        return [sorted(list(set(ss.lemma_names(lang=lang)) - {word})) for ss in self.synsets(word, lang=lang)]

    def doc(self, file='README', lang='eng'):
        """Return the contents of readme, license or citation file
        use lang=lang to get the file for an individual language"""
        if lang == 'eng':
            reader = self
        else:
            reader = self._omw_reader
            if lang in self.langs():
                file = f'{os.path.join(self.provenances[lang], file)}'
        try:
            with reader.open(file) as fp:
                return fp.read()
        except:
            if lang in self._lang_data:
                return f'Cannot determine {file} for {lang}'
            else:
                return f'Language {lang} is not supported.'

    def license(self, lang='eng'):
        """Return the contents of LICENSE (for omw)
        use lang=lang to get the license for an individual language"""
        return self.doc(file='LICENSE', lang=lang)

    def readme(self, lang='eng'):
        """Return the contents of README (for omw)
        use lang=lang to get the readme for an individual language"""
        return self.doc(file='README', lang=lang)

    def citation(self, lang='eng'):
        """Return the contents of citation.bib file (for omw)
        use lang=lang to get the citation for an individual language"""
        return self.doc(file='citation.bib', lang=lang)

    def lemma_count(self, lemma):
        """Return the frequency count for this Lemma"""
        if lemma._lang != 'eng':
            return 0
        if self._key_count_file is None:
            self._key_count_file = self.open('cntlist.rev')
        line = _binary_search_file(self._key_count_file, lemma._key)
        if line:
            return int(line.rsplit(' ', 1)[-1])
        else:
            return 0

    def path_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.path_similarity(synset2, verbose, simulate_root)
    path_similarity.__doc__ = Synset.path_similarity.__doc__

    def lch_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.lch_similarity(synset2, verbose, simulate_root)
    lch_similarity.__doc__ = Synset.lch_similarity.__doc__

    def wup_similarity(self, synset1, synset2, verbose=False, simulate_root=True):
        return synset1.wup_similarity(synset2, verbose, simulate_root)
    wup_similarity.__doc__ = Synset.wup_similarity.__doc__

    def res_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.res_similarity(synset2, ic, verbose)
    res_similarity.__doc__ = Synset.res_similarity.__doc__

    def jcn_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.jcn_similarity(synset2, ic, verbose)
    jcn_similarity.__doc__ = Synset.jcn_similarity.__doc__

    def lin_similarity(self, synset1, synset2, ic, verbose=False):
        return synset1.lin_similarity(synset2, ic, verbose)
    lin_similarity.__doc__ = Synset.lin_similarity.__doc__

    def morphy(self, form, pos=None, check_exceptions=True):
        """
        Find a possible base form for the given form, with the given
        part of speech, by checking WordNet's list of exceptional
        forms, and by recursively stripping affixes for this part of
        speech until a form in WordNet is found.

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.morphy('dogs'))
        dog
        >>> print(wn.morphy('churches'))
        church
        >>> print(wn.morphy('aardwolves'))
        aardwolf
        >>> print(wn.morphy('abaci'))
        abacus
        >>> wn.morphy('hardrock', wn.ADV)
        >>> print(wn.morphy('book', wn.NOUN))
        book
        >>> wn.morphy('book', wn.ADJ)
        """
        if pos is None:
            morphy = self._morphy
            analyses = chain((a for p in POS_LIST for a in morphy(form, p)))
        else:
            analyses = self._morphy(form, pos, check_exceptions)
        first = list(islice(analyses, 1))
        if len(first) == 1:
            return first[0]
        else:
            return None
    MORPHOLOGICAL_SUBSTITUTIONS = {NOUN: [('s', ''), ('ses', 's'), ('ves', 'f'), ('xes', 'x'), ('zes', 'z'), ('ches', 'ch'), ('shes', 'sh'), ('men', 'man'), ('ies', 'y')], VERB: [('s', ''), ('ies', 'y'), ('es', 'e'), ('es', ''), ('ed', 'e'), ('ed', ''), ('ing', 'e'), ('ing', '')], ADJ: [('er', ''), ('est', ''), ('er', 'e'), ('est', 'e')], ADV: []}
    MORPHOLOGICAL_SUBSTITUTIONS[ADJ_SAT] = MORPHOLOGICAL_SUBSTITUTIONS[ADJ]

    def _morphy(self, form, pos, check_exceptions=True):
        exceptions = self._exception_map[pos]
        substitutions = self.MORPHOLOGICAL_SUBSTITUTIONS[pos]

        def apply_rules(forms):
            return [form[:-len(old)] + new for form in forms for old, new in substitutions if form.endswith(old)]

        def filter_forms(forms):
            result = []
            seen = set()
            for form in forms:
                if form in self._lemma_pos_offset_map:
                    if pos in self._lemma_pos_offset_map[form]:
                        if form not in seen:
                            result.append(form)
                            seen.add(form)
            return result
        if check_exceptions:
            if form in exceptions:
                return filter_forms([form] + exceptions[form])
        forms = apply_rules([form])
        results = filter_forms([form] + forms)
        if results:
            return results
        while forms:
            forms = apply_rules(forms)
            results = filter_forms(forms)
            if results:
                return results
        return []

    def ic(self, corpus, weight_senses_equally=False, smoothing=1.0):
        """
        Creates an information content lookup dictionary from a corpus.

        :type corpus: CorpusReader
        :param corpus: The corpus from which we create an information
            content dictionary.
        :type weight_senses_equally: bool
        :param weight_senses_equally: If this is True, gives all
            possible senses equal weight rather than dividing by the
            number of possible senses.  (If a word has 3 synses, each
            sense gets 0.3333 per appearance when this is False, 1.0 when
            it is true.)
        :param smoothing: How much do we smooth synset counts (default is 1.0)
        :type smoothing: float
        :return: An information content dictionary
        """
        counts = FreqDist()
        for ww in corpus.words():
            counts[ww] += 1
        ic = {}
        for pp in POS_LIST:
            ic[pp] = defaultdict(float)
        if smoothing > 0.0:
            for pp in POS_LIST:
                ic[pp][0] = smoothing
            for ss in self.all_synsets():
                pos = ss._pos
                if pos == ADJ_SAT:
                    pos = ADJ
                ic[pos][ss._offset] = smoothing
        for ww in counts:
            possible_synsets = self.synsets(ww)
            if len(possible_synsets) == 0:
                continue
            weight = float(counts[ww])
            if not weight_senses_equally:
                weight /= float(len(possible_synsets))
            for ss in possible_synsets:
                pos = ss._pos
                if pos == ADJ_SAT:
                    pos = ADJ
                for level in ss._iter_hypernym_lists():
                    for hh in level:
                        ic[pos][hh._offset] += weight
                ic[pos][0] += weight
        return ic

    def custom_lemmas(self, tab_file, lang):
        """
        Reads a custom tab file containing mappings of lemmas in the given
        language to Princeton WordNet 3.0 synset offsets, allowing NLTK's
        WordNet functions to then be used with that language.

        See the "Tab files" section at https://omwn.org/omw1.html for
        documentation on the Multilingual WordNet tab file format.

        :param tab_file: Tab file as a file or file-like object
        :type: lang str
        :param: lang ISO 639-3 code of the language of the tab file
        """
        lg = lang.split('_')[0]
        if len(lg) != 3:
            raise ValueError('lang should be a (3 character) ISO 639-3 code')
        self._lang_data[lang] = [defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)]
        for line in tab_file.readlines():
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            if not line.startswith('#'):
                triple = line.strip().split('\t')
                if len(triple) < 3:
                    continue
                offset_pos, label = triple[:2]
                val = triple[-1]
                if self.map30:
                    if offset_pos in self.map30:
                        offset_pos = self.map30[offset_pos]
                    else:
                        if offset_pos not in self.nomap and offset_pos.replace('a', 's') not in self.nomap:
                            warnings.warn(f"{lang}: invalid offset {offset_pos} in '{line}'")
                        continue
                elif offset_pos[-1] == 'a':
                    wnss = self.of2ss(offset_pos)
                    if wnss and wnss.pos() == 's':
                        offset_pos = self.ss2of(wnss)
                pair = label.split(':')
                attr = pair[-1]
                if len(pair) == 1 or pair[0] == lg:
                    if attr == 'lemma':
                        val = val.strip().replace(' ', '_')
                        self._lang_data[lang][1][val.lower()].append(offset_pos)
                    if attr in self.lg_attrs:
                        self._lang_data[lang][self.lg_attrs.index(attr)][offset_pos].append(val)

    def disable_custom_lemmas(self, lang):
        """prevent synsets from being mistakenly added"""
        for n in range(len(self.lg_attrs)):
            self._lang_data[lang][n].default_factory = None

    def digraph(self, inputs, rel=lambda s: s.hypernyms(), pos=None, maxdepth=-1, shapes=None, attr=None, verbose=False):
        """
        Produce a graphical representation from 'inputs' (a list of
        start nodes, which can be a mix of Synsets, Lemmas and/or words),
        and a synset relation, for drawing with the 'dot' graph visualisation
        program from the Graphviz package.

        Return a string in the DOT graph file language, which can then be
        converted to an image by nltk.parse.dependencygraph.dot2img(dot_string).

        Optional Parameters:
        :rel: Wordnet synset relation
        :pos: for words, restricts Part of Speech to 'n', 'v', 'a' or 'r'
        :maxdepth: limit the longest path
        :shapes: dictionary of strings that trigger a specified shape
        :attr: dictionary with global graph attributes
        :verbose: warn about cycles

        >>> from nltk.corpus import wordnet as wn
        >>> print(wn.digraph([wn.synset('dog.n.01')]))
        digraph G {
        "Synset('animal.n.01')" -> "Synset('organism.n.01')";
        "Synset('canine.n.02')" -> "Synset('carnivore.n.01')";
        "Synset('carnivore.n.01')" -> "Synset('placental.n.01')";
        "Synset('chordate.n.01')" -> "Synset('animal.n.01')";
        "Synset('dog.n.01')" -> "Synset('canine.n.02')";
        "Synset('dog.n.01')" -> "Synset('domestic_animal.n.01')";
        "Synset('domestic_animal.n.01')" -> "Synset('animal.n.01')";
        "Synset('living_thing.n.01')" -> "Synset('whole.n.02')";
        "Synset('mammal.n.01')" -> "Synset('vertebrate.n.01')";
        "Synset('object.n.01')" -> "Synset('physical_entity.n.01')";
        "Synset('organism.n.01')" -> "Synset('living_thing.n.01')";
        "Synset('physical_entity.n.01')" -> "Synset('entity.n.01')";
        "Synset('placental.n.01')" -> "Synset('mammal.n.01')";
        "Synset('vertebrate.n.01')" -> "Synset('chordate.n.01')";
        "Synset('whole.n.02')" -> "Synset('object.n.01')";
        }
        <BLANKLINE>
        """
        from nltk.util import edge_closure, edges2dot
        synsets = set()
        edges = set()
        if not shapes:
            shapes = dict()
        if not attr:
            attr = dict()

        def add_lemma(lem):
            ss = lem.synset()
            synsets.add(ss)
            edges.add((lem, ss))
        for node in inputs:
            typ = type(node)
            if typ == Synset:
                synsets.add(node)
            elif typ == Lemma:
                add_lemma(node)
            elif typ == str:
                for lemma in self.lemmas(node, pos):
                    add_lemma(lemma)
        for ss in synsets:
            edges = edges.union(edge_closure(ss, rel, maxdepth, verbose))
        dot_string = edges2dot(sorted(list(edges)), shapes=shapes, attr=attr)
        return dot_string