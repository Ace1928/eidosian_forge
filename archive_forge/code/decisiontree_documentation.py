from collections import defaultdict
from nltk.classify.api import ClassifierI
from nltk.probability import FreqDist, MLEProbDist, entropy

        :param binary: If true, then treat all feature/value pairs as
            individual binary features, rather than using a single n-way
            branch for each feature.
        