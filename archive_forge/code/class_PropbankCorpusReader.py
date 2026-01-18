import re
from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class PropbankCorpusReader(CorpusReader):
    """
    Corpus reader for the propbank corpus, which augments the Penn
    Treebank with information about the predicate argument structure
    of every verb instance.  The corpus consists of two parts: the
    predicate-argument annotations themselves, and a set of "frameset
    files" which define the argument labels used by the annotations,
    on a per-verb basis.  Each "frameset file" contains one or more
    predicates, such as ``'turn'`` or ``'turn_on'``, each of which is
    divided into coarse-grained word senses called "rolesets".  For
    each "roleset", the frameset file provides descriptions of the
    argument roles, along with examples.
    """

    def __init__(self, root, propfile, framefiles='', verbsfile=None, parse_fileid_xform=None, parse_corpus=None, encoding='utf8'):
        """
        :param root: The root directory for this corpus.
        :param propfile: The name of the file containing the predicate-
            argument annotations (relative to ``root``).
        :param framefiles: A list or regexp specifying the frameset
            fileids for this corpus.
        :param parse_fileid_xform: A transform that should be applied
            to the fileids in this corpus.  This should be a function
            of one argument (a fileid) that returns a string (the new
            fileid).
        :param parse_corpus: The corpus containing the parse trees
            corresponding to this corpus.  These parse trees are
            necessary to resolve the tree pointers used by propbank.
        """
        if isinstance(framefiles, str):
            framefiles = find_corpus_fileids(root, framefiles)
        framefiles = list(framefiles)
        CorpusReader.__init__(self, root, [propfile, verbsfile] + framefiles, encoding)
        self._propfile = propfile
        self._framefiles = framefiles
        self._verbsfile = verbsfile
        self._parse_fileid_xform = parse_fileid_xform
        self._parse_corpus = parse_corpus

    def instances(self, baseform=None):
        """
        :return: a corpus view that acts as a list of
            ``PropBankInstance`` objects, one for each noun in the corpus.
        """
        kwargs = {}
        if baseform is not None:
            kwargs['instance_filter'] = lambda inst: inst.baseform == baseform
        return StreamBackedCorpusView(self.abspath(self._propfile), lambda stream: self._read_instance_block(stream, **kwargs), encoding=self.encoding(self._propfile))

    def lines(self):
        """
        :return: a corpus view that acts as a list of strings, one for
            each line in the predicate-argument annotation file.
        """
        return StreamBackedCorpusView(self.abspath(self._propfile), read_line_block, encoding=self.encoding(self._propfile))

    def roleset(self, roleset_id):
        """
        :return: the xml description for the given roleset.
        """
        baseform = roleset_id.split('.')[0]
        framefile = 'frames/%s.xml' % baseform
        if framefile not in self._framefiles:
            raise ValueError('Frameset file for %s not found' % roleset_id)
        with self.abspath(framefile).open() as fp:
            etree = ElementTree.parse(fp).getroot()
        for roleset in etree.findall('predicate/roleset'):
            if roleset.attrib['id'] == roleset_id:
                return roleset
        raise ValueError(f'Roleset {roleset_id} not found in {framefile}')

    def rolesets(self, baseform=None):
        """
        :return: list of xml descriptions for rolesets.
        """
        if baseform is not None:
            framefile = 'frames/%s.xml' % baseform
            if framefile not in self._framefiles:
                raise ValueError('Frameset file for %s not found' % baseform)
            framefiles = [framefile]
        else:
            framefiles = self._framefiles
        rsets = []
        for framefile in framefiles:
            with self.abspath(framefile).open() as fp:
                etree = ElementTree.parse(fp).getroot()
            rsets.append(etree.findall('predicate/roleset'))
        return LazyConcatenation(rsets)

    def verbs(self):
        """
        :return: a corpus view that acts as a list of all verb lemmas
            in this corpus (from the verbs.txt file).
        """
        return StreamBackedCorpusView(self.abspath(self._verbsfile), read_line_block, encoding=self.encoding(self._verbsfile))

    def _read_instance_block(self, stream, instance_filter=lambda inst: True):
        block = []
        for i in range(100):
            line = stream.readline().strip()
            if line:
                inst = PropbankInstance.parse(line, self._parse_fileid_xform, self._parse_corpus)
                if instance_filter(inst):
                    block.append(inst)
        return block