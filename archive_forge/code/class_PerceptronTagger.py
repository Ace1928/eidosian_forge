import logging
import pickle
import random
from collections import defaultdict
from nltk import jsontags
from nltk.data import find, load
from nltk.tag.api import TaggerI
@jsontags.register_tag
class PerceptronTagger(TaggerI):
    """
    Greedy Averaged Perceptron tagger, as implemented by Matthew Honnibal.
    See more implementation details here:
    https://explosion.ai/blog/part-of-speech-pos-tagger-in-python

    >>> from nltk.tag.perceptron import PerceptronTagger

    Train the model

    >>> tagger = PerceptronTagger(load=False)

    >>> tagger.train([[('today','NN'),('is','VBZ'),('good','JJ'),('day','NN')],
    ... [('yes','NNS'),('it','PRP'),('beautiful','JJ')]])

    >>> tagger.tag(['today','is','a','beautiful','day'])
    [('today', 'NN'), ('is', 'PRP'), ('a', 'PRP'), ('beautiful', 'JJ'), ('day', 'NN')]

    Use the pretrain model (the default constructor)

    >>> pretrain = PerceptronTagger()

    >>> pretrain.tag('The quick brown fox jumps over the lazy dog'.split())
    [('The', 'DT'), ('quick', 'JJ'), ('brown', 'NN'), ('fox', 'NN'), ('jumps', 'VBZ'), ('over', 'IN'), ('the', 'DT'), ('lazy', 'JJ'), ('dog', 'NN')]

    >>> pretrain.tag("The red cat".split())
    [('The', 'DT'), ('red', 'JJ'), ('cat', 'NN')]
    """
    json_tag = 'nltk.tag.sequential.PerceptronTagger'
    START = ['-START-', '-START2-']
    END = ['-END-', '-END2-']

    def __init__(self, load=True):
        """
        :param load: Load the pickled model upon instantiation.
        """
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            AP_MODEL_LOC = 'file:' + str(find('taggers/averaged_perceptron_tagger/' + PICKLE))
            self.load(AP_MODEL_LOC)

    def tag(self, tokens, return_conf=False, use_tagdict=True):
        """
        Tag tokenized sentences.
        :params tokens: list of word
        :type tokens: list(str)
        """
        prev, prev2 = self.START
        output = []
        context = self.START + [self.normalize(w) for w in tokens] + self.END
        for i, word in enumerate(tokens):
            tag, conf = (self.tagdict.get(word), 1.0) if use_tagdict == True else (None, None)
            if not tag:
                features = self._get_features(i, word, context, prev, prev2)
                tag, conf = self.model.predict(features, return_conf)
            output.append((word, tag, conf) if return_conf == True else (word, tag))
            prev2 = prev
            prev = tag
        return output

    def train(self, sentences, save_loc=None, nr_iter=5):
        """Train a model from sentences, and save it at ``save_loc``. ``nr_iter``
        controls the number of Perceptron training iterations.

        :param sentences: A list or iterator of sentences, where each sentence
            is a list of (words, tags) tuples.
        :param save_loc: If not ``None``, saves a pickled model in this location.
        :param nr_iter: Number of training iterations.
        """
        self._sentences = list()
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter_ in range(nr_iter):
            c = 0
            n = 0
            for sentence in self._sentences:
                words, tags = zip(*sentence)
                prev, prev2 = self.START
                context = self.START + [self.normalize(w) for w in words] + self.END
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if not guess:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess, _ = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2 = prev
                    prev = guess
                    c += guess == tags[i]
                    n += 1
            random.shuffle(self._sentences)
            logging.info(f'Iter {iter_}: {c}/{n}={_pc(c, n)}')
        self._sentences = None
        self.model.average_weights()
        if save_loc is not None:
            with open(save_loc, 'wb') as fout:
                pickle.dump((self.model.weights, self.tagdict, self.classes), fout, 2)

    def load(self, loc):
        """
        :param loc: Load a pickled model at location.
        :type loc: str
        """
        self.model.weights, self.tagdict, self.classes = load(loc)
        self.model.classes = self.classes

    def encode_json_obj(self):
        return (self.model.weights, self.tagdict, list(self.classes))

    @classmethod
    def decode_json_obj(cls, obj):
        tagger = cls(load=False)
        tagger.model.weights, tagger.tagdict, tagger.classes = obj
        tagger.classes = set(tagger.classes)
        tagger.model.classes = tagger.classes
        return tagger

    def normalize(self, word):
        """
        Normalization used in pre-processing.
        - All words are lower cased
        - Groups of digits of length 4 are represented as !YEAR;
        - Other digits are represented as !DIGITS

        :rtype: str
        """
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        if word.isdigit() and len(word) == 4:
            return '!YEAR'
        if word and word[0].isdigit():
            return '!DIGITS'
        return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        """Map tokens into a feature representation, implemented as a
        {hashable: int} dict. If the features change, a new model must be
        trained.
        """

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1
        i += len(self.START)
        features = defaultdict(int)
        add('bias')
        add('i suffix', word[-3:])
        add('i pref1', word[0] if word else '')
        add('i-1 tag', prev)
        add('i-2 tag', prev2)
        add('i tag+i-2 tag', prev, prev2)
        add('i word', context[i])
        add('i-1 tag+i word', prev, context[i])
        add('i-1 word', context[i - 1])
        add('i-1 suffix', context[i - 1][-3:])
        add('i-2 word', context[i - 2])
        add('i+1 word', context[i + 1])
        add('i+1 suffix', context[i + 1][-3:])
        add('i+2 word', context[i + 2])
        return features

    def _make_tagdict(self, sentences):
        """
        Make a tag dictionary for single-tag words.
        :param sentences: A list of list of (word, tag) tuples.
        """
        counts = defaultdict(lambda: defaultdict(int))
        for sentence in sentences:
            self._sentences.append(sentence)
            for word, tag in sentence:
                counts[word][tag] += 1
                self.classes.add(tag)
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            n = sum(tag_freqs.values())
            if n >= freq_thresh and mode / n >= ambiguity_thresh:
                self.tagdict[word] = tag