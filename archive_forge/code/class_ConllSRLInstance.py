import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
class ConllSRLInstance:
    """
    An SRL instance from a CoNLL corpus, which identifies and
    providing labels for the arguments of a single verb.
    """

    def __init__(self, tree, verb_head, verb_stem, roleset, tagged_spans):
        self.verb = []
        "A list of the word indices of the words that compose the\n           verb whose arguments are identified by this instance.\n           This will contain multiple word indices when multi-word\n           verbs are used (e.g. 'turn on')."
        self.verb_head = verb_head
        "The word index of the head word of the verb whose arguments\n           are identified by this instance.  E.g., for a sentence that\n           uses the verb 'turn on,' ``verb_head`` will be the word index\n           of the word 'turn'."
        self.verb_stem = verb_stem
        self.roleset = roleset
        self.arguments = []
        'A list of ``(argspan, argid)`` tuples, specifying the location\n           and type for each of the arguments identified by this\n           instance.  ``argspan`` is a tuple ``start, end``, indicating\n           that the argument consists of the ``words[start:end]``.'
        self.tagged_spans = tagged_spans
        'A list of ``(span, id)`` tuples, specifying the location and\n           type for each of the arguments, as well as the verb pieces,\n           that make up this instance.'
        self.tree = tree
        'The parse tree for the sentence containing this instance.'
        self.words = tree.leaves()
        'A list of the words in the sentence containing this\n           instance.'
        for (start, end), tag in tagged_spans:
            if tag in ('V', 'C-V'):
                self.verb += list(range(start, end))
            else:
                self.arguments.append(((start, end), tag))

    def __repr__(self):
        plural = 's' if len(self.arguments) != 1 else ''
        return '<ConllSRLInstance for %r with %d argument%s>' % (self.verb_stem, len(self.arguments), plural)

    def pprint(self):
        verbstr = ' '.join((self.words[i][0] for i in self.verb))
        hdr = f'SRL for {verbstr!r} (stem={self.verb_stem!r}):\n'
        s = ''
        for i, word in enumerate(self.words):
            if isinstance(word, tuple):
                word = word[0]
            for (start, end), argid in self.arguments:
                if i == start:
                    s += '[%s ' % argid
                if i == end:
                    s += '] '
            if i in self.verb:
                word = '<<%s>>' % word
            s += word + ' '
        return hdr + textwrap.fill(s.replace(' ]', ']'), initial_indent='    ', subsequent_indent='    ')