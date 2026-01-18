import numpy as np
from . import check_consistent_length, check_matplotlib_support
from ._response import _get_response_values_binary
from .multiclass import type_of_target
from .validation import _check_pos_label_consistency
def _validate_score_name(score_name, scoring, negate_score):
    """Validate the `score_name` parameter.

    If `score_name` is provided, we just return it as-is.
    If `score_name` is `None`, we use `Score` if `negate_score` is `False` and
    `Negative score` otherwise.
    If `score_name` is a string or a callable, we infer the name. We replace `_` by
    spaces and capitalize the first letter. We remove `neg_` and replace it by
    `"Negative"` if `negate_score` is `False` or just remove it otherwise.
    """
    if score_name is not None:
        return score_name
    elif scoring is None:
        return 'Negative score' if negate_score else 'Score'
    else:
        score_name = scoring.__name__ if callable(scoring) else scoring
        if negate_score:
            if score_name.startswith('neg_'):
                score_name = score_name[4:]
            else:
                score_name = f'Negative {score_name}'
        elif score_name.startswith('neg_'):
            score_name = f'Negative {score_name[4:]}'
        score_name = score_name.replace('_', ' ')
        return score_name.capitalize()