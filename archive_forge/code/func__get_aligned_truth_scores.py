from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from thinc.api import Config, Model, Optimizer, set_dropout_rate
from thinc.types import Floats2d
from ..errors import Errors
from ..language import Language
from ..scorer import Scorer
from ..tokens import Doc, Span
from ..training import Example
from ..util import registry
from .spancat import DEFAULT_SPANS_KEY
from .trainable_pipe import TrainablePipe
def _get_aligned_truth_scores(self, examples, ops) -> Tuple[Floats2d, Floats2d]:
    """Align scores of the predictions to the references for calculating
        the loss.
        """
    truths = []
    masks = []
    for eg in examples:
        if eg.x.text != eg.y.text:
            raise ValueError(Errors.E1054.format(component='span_finder'))
        n_tokens = len(eg.predicted)
        truth = ops.xp.zeros((n_tokens, 2), dtype='float32')
        mask = ops.xp.ones((n_tokens, 2), dtype='float32')
        if self.cfg['spans_key'] in eg.reference.spans:
            for span in eg.reference.spans[self.cfg['spans_key']]:
                ref_start_char, ref_end_char = _char_indices(span)
                pred_span = eg.predicted.char_span(ref_start_char, ref_end_char, alignment_mode='expand')
                pred_start_char, pred_end_char = _char_indices(pred_span)
                start_match = pred_start_char == ref_start_char
                end_match = pred_end_char == ref_end_char
                if start_match:
                    truth[pred_span[0].i, 0] = 1
                else:
                    mask[pred_span[0].i, 0] = 0
                if end_match:
                    truth[pred_span[-1].i, 1] = 1
                else:
                    mask[pred_span[-1].i, 1] = 0
        truths.append(truth)
        masks.append(mask)
    truths = ops.xp.concatenate(truths, axis=0)
    masks = ops.xp.concatenate(masks, axis=0)
    return (truths, masks)