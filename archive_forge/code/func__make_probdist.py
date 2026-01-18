from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist
def _make_probdist(self, y_proba):
    classes = self._encoder.classes_
    return DictionaryProbDist({classes[i]: p for i, p in enumerate(y_proba)})