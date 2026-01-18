from collections import Counter
from . import EvaluationMixin, evaluation_io
from ..io import load_key
class KeyEvaluation(EvaluationMixin):
    """
    Provide the key evaluation score.

    Parameters
    ----------
    detection : str
        File containing detected key
    annotation : str
        File containing annotated key
    strict_fifth : bool, optional
        Use strict interpretation of the 'fifth' category, as in MIREX.
    name : str, optional
        Name of the evaluation object (e.g., the name of the song).

    """
    METRIC_NAMES = [('score', 'Score'), ('error_category', 'Error Category')]

    def __init__(self, detection, annotation, strict_fifth=False, name=None, **kwargs):
        self.name = name or ''
        self.detection = key_label_to_class(detection)
        self.annotation = key_label_to_class(annotation)
        self.score, self.error_category = error_type(self.detection, self.annotation, strict_fifth)

    def tostring(self, **kwargs):
        """
        Format the evaluation as a human readable string.

        Returns
        -------
        str
            Evaluation score and category as a human readable string.

        """
        ret = '{}: '.format(self.name) if self.name else ''
        ret += '{:3.1f}, {}'.format(self.score, self.error_category)
        return ret