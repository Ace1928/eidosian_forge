import enum
import inspect
from typing import Iterable, List, Optional, Tuple, Union
def _align_output_features_output_indices(out_features: Optional[List[str]], out_indices: Optional[Union[List[int], Tuple[int]]], stage_names: List[str]):
    """
    Finds the corresponding `out_features` and `out_indices` for the given `stage_names`.

    The logic is as follows:
        - `out_features` not set, `out_indices` set: `out_features` is set to the `out_features` corresponding to the
        `out_indices`.
        - `out_indices` not set, `out_features` set: `out_indices` is set to the `out_indices` corresponding to the
        `out_features`.
        - `out_indices` and `out_features` not set: `out_indices` and `out_features` are set to the last stage.
        - `out_indices` and `out_features` set: input `out_indices` and `out_features` are returned.

    Args:
        out_features (`List[str]`): The names of the features for the backbone to output.
        out_indices (`List[int]` or `Tuple[int]`): The indices of the features for the backbone to output.
        stage_names (`List[str]`): The names of the stages of the backbone.
    """
    if out_indices is None and out_features is None:
        out_indices = [len(stage_names) - 1]
        out_features = [stage_names[-1]]
    elif out_indices is None and out_features is not None:
        out_indices = [stage_names.index(layer) for layer in out_features]
    elif out_features is None and out_indices is not None:
        out_features = [stage_names[idx] for idx in out_indices]
    return (out_features, out_indices)