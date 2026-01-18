import numpy as np
from . import evaluation_io, EvaluationMixin
from ..io import load_chords
class ChordEvaluation(EvaluationMixin):
    """
    Provide various chord evaluation scores.

    Parameters
    ----------
    detections : str
        File containing chords detections.
    annotations : str
        File containing chord annotations.
    name : str, optional
        Name of the evaluation object (e.g., the name of the song).

    """
    METRIC_NAMES = [('root', 'Root'), ('majmin', 'MajMin'), ('majminbass', 'MajMinBass'), ('sevenths', 'Sevenths'), ('seventhsbass', 'SeventhsBass'), ('segmentation', 'Segmentation'), ('oversegmentation', 'OverSegmentation'), ('undersegmentation', 'UnderSegmentation')]

    def __init__(self, detections, annotations, name=None, **kwargs):
        self.name = name or ''
        self.ann_chords = merge_chords(encode(annotations))
        self.det_chords = merge_chords(adjust(encode(detections), self.ann_chords))
        self.annotations, self.detections, self.durations = evaluation_pairs(self.det_chords, self.ann_chords)
        self._underseg = None
        self._overseg = None

    @property
    def length(self):
        """Length of annotations."""
        return self.ann_chords['end'][-1] - self.ann_chords['start'][0]

    @property
    def root(self):
        """Fraction of correctly detected chord roots."""
        return np.average(score_root(self.detections, self.annotations), weights=self.durations)

    @property
    def majmin(self):
        """
        Fraction of correctly detected chords that can be reduced to major
        or minor triads (plus no-chord). Ignores the bass pitch class.
        """
        det_triads = reduce_to_triads(self.detections)
        ann_triads = reduce_to_triads(self.annotations)
        majmin_sel = select_majmin(ann_triads)
        return np.average(score_exact(det_triads, ann_triads), weights=self.durations * majmin_sel)

    @property
    def majminbass(self):
        """
        Fraction of correctly detected chords that can be reduced to major
        or minor triads (plus no-chord). Considers the bass pitch class.
        """
        det_triads = reduce_to_triads(self.detections, keep_bass=True)
        ann_triads = reduce_to_triads(self.annotations, keep_bass=True)
        majmin_sel = select_majmin(ann_triads)
        return np.average(score_exact(det_triads, ann_triads), weights=self.durations * majmin_sel)

    @property
    def sevenths(self):
        """
        Fraction of correctly detected chords that can be reduced to a seventh
        tetrad (plus no-chord). Ignores the bass pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections)
        ann_tetrads = reduce_to_tetrads(self.annotations)
        sevenths_sel = select_sevenths(ann_tetrads)
        return np.average(score_exact(det_tetrads, ann_tetrads), weights=self.durations * sevenths_sel)

    @property
    def seventhsbass(self):
        """
        Fraction of correctly detected chords that can be reduced to a seventh
        tetrad (plus no-chord). Considers the bass pitch class.
        """
        det_tetrads = reduce_to_tetrads(self.detections, keep_bass=True)
        ann_tetrads = reduce_to_tetrads(self.annotations, keep_bass=True)
        sevenths_sel = select_sevenths(ann_tetrads)
        return np.average(score_exact(det_tetrads, ann_tetrads), weights=self.durations * sevenths_sel)

    @property
    def undersegmentation(self):
        """
        Normalized Hamming divergence (directional) between annotations and
        detections. Captures missed chord segments.
        """
        if self._underseg is None:
            self._underseg = 1 - segmentation(self.det_chords['start'], self.det_chords['end'], self.ann_chords['start'], self.ann_chords['end'])
        return self._underseg

    @property
    def oversegmentation(self):
        """
        Normalized Hamming divergence (directional) between detections and
        annotations. Captures how fragmented the detected chord segments are.
        """
        if self._overseg is None:
            self._overseg = 1 - segmentation(self.ann_chords['start'], self.ann_chords['end'], self.det_chords['start'], self.det_chords['end'])
        return self._overseg

    @property
    def segmentation(self):
        """Minimum of `oversegmentation` and `undersegmentation`."""
        return min(self.undersegmentation, self.oversegmentation)

    def tostring(self, **kwargs):
        """
        Format the evaluation metrics as a human readable string.

        Returns
        -------
        eval_string : str
            Evaluation metrics formatted as a human readable string.

        """
        ret = '{}\n  Root: {:5.2f} MajMin: {:5.2f} MajMinBass: {:5.2f} Sevenths: {:5.2f} SeventhsBass: {:5.2f}\n  Seg: {:5.2f} UnderSeg: {:5.2f} OverSeg: {:5.2f}'.format(self.name, self.root * 100, self.majmin * 100, self.majminbass * 100, self.sevenths * 100, self.seventhsbass * 100, self.segmentation * 100, self.undersegmentation * 100, self.oversegmentation * 100)
        return ret