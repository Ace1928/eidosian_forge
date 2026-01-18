import os
import re
import subprocess
import tempfile
import time
import zipfile
from sys import stdin
from nltk.classify.api import ClassifierI
from nltk.internals import config_java, java
from nltk.probability import DictionaryProbDist
class ARFF_Formatter:
    """
    Converts featuresets and labeled featuresets to ARFF-formatted
    strings, appropriate for input into Weka.

    Features and classes can be specified manually in the constructor, or may
    be determined from data using ``from_train``.
    """

    def __init__(self, labels, features):
        """
        :param labels: A list of all class labels that can be generated.
        :param features: A list of feature specifications, where
            each feature specification is a tuple (fname, ftype);
            and ftype is an ARFF type string such as NUMERIC or
            STRING.
        """
        self._labels = labels
        self._features = features

    def format(self, tokens):
        """Returns a string representation of ARFF output for the given data."""
        return self.header_section() + self.data_section(tokens)

    def labels(self):
        """Returns the list of classes."""
        return list(self._labels)

    def write(self, outfile, tokens):
        """Writes ARFF data to a file for the given data."""
        if not hasattr(outfile, 'write'):
            outfile = open(outfile, 'w')
        outfile.write(self.format(tokens))
        outfile.close()

    @staticmethod
    def from_train(tokens):
        """
        Constructs an ARFF_Formatter instance with class labels and feature
        types determined from the given data. Handles boolean, numeric and
        string (note: not nominal) types.
        """
        labels = {label for tok, label in tokens}
        features = {}
        for tok, label in tokens:
            for fname, fval in tok.items():
                if issubclass(type(fval), bool):
                    ftype = '{True, False}'
                elif issubclass(type(fval), (int, float, bool)):
                    ftype = 'NUMERIC'
                elif issubclass(type(fval), str):
                    ftype = 'STRING'
                elif fval is None:
                    continue
                else:
                    raise ValueError('Unsupported value type %r' % ftype)
                if features.get(fname, ftype) != ftype:
                    raise ValueError('Inconsistent type for %s' % fname)
                features[fname] = ftype
        features = sorted(features.items())
        return ARFF_Formatter(labels, features)

    def header_section(self):
        """Returns an ARFF header as a string."""
        s = '% Weka ARFF file\n' + '% Generated automatically by NLTK\n' + '%% %s\n\n' % time.ctime()
        s += '@RELATION rel\n\n'
        for fname, ftype in self._features:
            s += '@ATTRIBUTE %-30r %s\n' % (fname, ftype)
        s += '@ATTRIBUTE %-30r {%s}\n' % ('-label-', ','.join(self._labels))
        return s

    def data_section(self, tokens, labeled=None):
        """
        Returns the ARFF data section for the given data.

        :param tokens: a list of featuresets (dicts) or labelled featuresets
            which are tuples (featureset, label).
        :param labeled: Indicates whether the given tokens are labeled
            or not.  If None, then the tokens will be assumed to be
            labeled if the first token's value is a tuple or list.
        """
        if labeled is None:
            labeled = tokens and isinstance(tokens[0], (tuple, list))
        if not labeled:
            tokens = [(tok, None) for tok in tokens]
        s = '\n@DATA\n'
        for tok, label in tokens:
            for fname, ftype in self._features:
                s += '%s,' % self._fmt_arff_val(tok.get(fname))
            s += '%s\n' % self._fmt_arff_val(label)
        return s

    def _fmt_arff_val(self, fval):
        if fval is None:
            return '?'
        elif isinstance(fval, (bool, int)):
            return '%s' % fval
        elif isinstance(fval, float):
            return '%r' % fval
        else:
            return '%r' % fval