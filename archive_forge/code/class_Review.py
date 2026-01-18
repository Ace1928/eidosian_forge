import re
from nltk.corpus.reader.api import *
from nltk.tokenize import *
class Review:
    """
    A Review is the main block of a ReviewsCorpusReader.
    """

    def __init__(self, title=None, review_lines=None):
        """
        :param title: the title of the review.
        :param review_lines: the list of the ReviewLines that belong to the Review.
        """
        self.title = title
        if review_lines is None:
            self.review_lines = []
        else:
            self.review_lines = review_lines

    def add_line(self, review_line):
        """
        Add a line (ReviewLine) to the review.

        :param review_line: a ReviewLine instance that belongs to the Review.
        """
        assert isinstance(review_line, ReviewLine)
        self.review_lines.append(review_line)

    def features(self):
        """
        Return a list of features in the review. Each feature is a tuple made of
        the specific item feature and the opinion strength about that feature.

        :return: all features of the review as a list of tuples (feat, score).
        :rtype: list(tuple)
        """
        features = []
        for review_line in self.review_lines:
            features.extend(review_line.features)
        return features

    def sents(self):
        """
        Return all tokenized sentences in the review.

        :return: all sentences of the review as lists of tokens.
        :rtype: list(list(str))
        """
        return [review_line.sent for review_line in self.review_lines]

    def __repr__(self):
        return 'Review(title="{}", review_lines={})'.format(self.title, self.review_lines)