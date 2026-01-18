from collections import defaultdict
from nltk.classify.naivebayes import NaiveBayesClassifier
from nltk.probability import DictionaryProbDist, ELEProbDist, FreqDist
class PositiveNaiveBayesClassifier(NaiveBayesClassifier):

    @staticmethod
    def train(positive_featuresets, unlabeled_featuresets, positive_prob_prior=0.5, estimator=ELEProbDist):
        """
        :param positive_featuresets: An iterable of featuresets that are known as positive
            examples (i.e., their label is ``True``).

        :param unlabeled_featuresets: An iterable of featuresets whose label is unknown.

        :param positive_prob_prior: A prior estimate of the probability of the label
            ``True`` (default 0.5).
        """
        positive_feature_freqdist = defaultdict(FreqDist)
        unlabeled_feature_freqdist = defaultdict(FreqDist)
        feature_values = defaultdict(set)
        fnames = set()
        num_positive_examples = 0
        for featureset in positive_featuresets:
            for fname, fval in featureset.items():
                positive_feature_freqdist[fname][fval] += 1
                feature_values[fname].add(fval)
                fnames.add(fname)
            num_positive_examples += 1
        num_unlabeled_examples = 0
        for featureset in unlabeled_featuresets:
            for fname, fval in featureset.items():
                unlabeled_feature_freqdist[fname][fval] += 1
                feature_values[fname].add(fval)
                fnames.add(fname)
            num_unlabeled_examples += 1
        for fname in fnames:
            count = positive_feature_freqdist[fname].N()
            positive_feature_freqdist[fname][None] += num_positive_examples - count
            feature_values[fname].add(None)
        for fname in fnames:
            count = unlabeled_feature_freqdist[fname].N()
            unlabeled_feature_freqdist[fname][None] += num_unlabeled_examples - count
            feature_values[fname].add(None)
        negative_prob_prior = 1.0 - positive_prob_prior
        label_probdist = DictionaryProbDist({True: positive_prob_prior, False: negative_prob_prior})
        feature_probdist = {}
        for fname, freqdist in positive_feature_freqdist.items():
            probdist = estimator(freqdist, bins=len(feature_values[fname]))
            feature_probdist[True, fname] = probdist
        for fname, freqdist in unlabeled_feature_freqdist.items():
            global_probdist = estimator(freqdist, bins=len(feature_values[fname]))
            negative_feature_probs = {}
            for fval in feature_values[fname]:
                prob = (global_probdist.prob(fval) - positive_prob_prior * feature_probdist[True, fname].prob(fval)) / negative_prob_prior
                negative_feature_probs[fval] = max(prob, 0.0)
            feature_probdist[False, fname] = DictionaryProbDist(negative_feature_probs, normalize=True)
        return PositiveNaiveBayesClassifier(label_probdist, feature_probdist)