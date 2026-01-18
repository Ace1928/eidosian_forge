from functools import total_ordering
from xml.etree import ElementTree
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.internals import raise_unorderable_types
from nltk.tree import Tree
class NombankInstance:

    def __init__(self, fileid, sentnum, wordnum, baseform, sensenumber, predicate, predid, arguments, parse_corpus=None):
        self.fileid = fileid
        "The name of the file containing the parse tree for this\n        instance's sentence."
        self.sentnum = sentnum
        'The sentence number of this sentence within ``fileid``.\n        Indexing starts from zero.'
        self.wordnum = wordnum
        "The word number of this instance's predicate within its\n        containing sentence.  Word numbers are indexed starting from\n        zero, and include traces and other empty parse elements."
        self.baseform = baseform
        'The baseform of the predicate.'
        self.sensenumber = sensenumber
        'The sense number of the predicate.'
        self.predicate = predicate
        "A ``NombankTreePointer`` indicating the position of this\n        instance's predicate within its containing sentence."
        self.predid = predid
        'Identifier of the predicate.'
        self.arguments = tuple(arguments)
        "A list of tuples (argloc, argid), specifying the location\n        and identifier for each of the predicate's argument in the\n        containing sentence.  Argument identifiers are strings such as\n        ``'ARG0'`` or ``'ARGM-TMP'``.  This list does *not* contain\n        the predicate."
        self.parse_corpus = parse_corpus
        'A corpus reader for the parse trees corresponding to the\n        instances in this nombank corpus.'

    @property
    def roleset(self):
        """The name of the roleset used by this instance's predicate.
        Use ``nombank.roleset() <NombankCorpusReader.roleset>`` to
        look up information about the roleset."""
        r = self.baseform.replace('%', 'perc-sign')
        r = r.replace('1/10', '1-slash-10').replace('1-slash-10', 'oneslashonezero')
        return f'{r}.{self.sensenumber}'

    def __repr__(self):
        return '<NombankInstance: {}, sent {}, word {}>'.format(self.fileid, self.sentnum, self.wordnum)

    def __str__(self):
        s = '{} {} {} {} {}'.format(self.fileid, self.sentnum, self.wordnum, self.baseform, self.sensenumber)
        items = self.arguments + ((self.predicate, 'rel'),)
        for argloc, argid in sorted(items):
            s += f' {argloc}-{argid}'
        return s

    def _get_tree(self):
        if self.parse_corpus is None:
            return None
        if self.fileid not in self.parse_corpus.fileids():
            return None
        return self.parse_corpus.parsed_sents(self.fileid)[self.sentnum]
    tree = property(_get_tree, doc='\n        The parse tree corresponding to this instance, or None if\n        the corresponding tree is not available.')

    @staticmethod
    def parse(s, parse_fileid_xform=None, parse_corpus=None):
        pieces = s.split()
        if len(pieces) < 6:
            raise ValueError('Badly formatted nombank line: %r' % s)
        fileid, sentnum, wordnum, baseform, sensenumber = pieces[:5]
        args = pieces[5:]
        rel = [args.pop(i) for i, p in enumerate(args) if '-rel' in p]
        if len(rel) != 1:
            raise ValueError('Badly formatted nombank line: %r' % s)
        if parse_fileid_xform is not None:
            fileid = parse_fileid_xform(fileid)
        sentnum = int(sentnum)
        wordnum = int(wordnum)
        predloc, predid = rel[0].split('-', 1)
        predicate = NombankTreePointer.parse(predloc)
        arguments = []
        for arg in args:
            argloc, argid = arg.split('-', 1)
            arguments.append((NombankTreePointer.parse(argloc), argid))
        return NombankInstance(fileid, sentnum, wordnum, baseform, sensenumber, predicate, predid, arguments, parse_corpus)