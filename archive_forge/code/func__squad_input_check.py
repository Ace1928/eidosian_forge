import re
import string
from collections import Counter
from typing import Any, Callable, Dict, List, Tuple, Union
from torch import Tensor, tensor
from torchmetrics.utilities import rank_zero_warn
def _squad_input_check(preds: PREDS_TYPE, targets: TARGETS_TYPE) -> Tuple[Dict[str, str], List[Dict[str, List[Dict[str, List[Dict[str, Any]]]]]]]:
    """Check for types and convert the input to necessary format to compute the input."""
    if isinstance(preds, Dict):
        preds = [preds]
    if isinstance(targets, Dict):
        targets = [targets]
    for pred in preds:
        pred_keys = pred.keys()
        if 'prediction_text' not in pred_keys or 'id' not in pred_keys:
            raise KeyError("Expected keys in a single prediction are 'prediction_text' and 'id'.Please make sure that 'prediction_text' maps to the answer string and 'id' maps to the key string.")
    for target in targets:
        target_keys = target.keys()
        if 'answers' not in target_keys or 'id' not in target_keys:
            raise KeyError(f"Expected keys in a single target are 'answers' and 'id'.Please make sure that 'answers' maps to a `SQuAD` format dictionary and 'id' maps to the key string.\nSQuAD Format: {SQuAD_FORMAT}")
        answers: Dict[str, Union[List[str], List[int]]] = target['answers']
        if 'text' not in answers:
            raise KeyError(f"Expected keys in a 'answers' are 'text'.Please make sure that 'answer' maps to a `SQuAD` format dictionary.\nSQuAD Format: {SQuAD_FORMAT}")
    preds_dict = {prediction['id']: prediction['prediction_text'] for prediction in preds}
    _fn_answer = lambda tgt: {'answers': [{'text': txt} for txt in tgt['answers']['text']], 'id': tgt['id']}
    targets_dict = [{'paragraphs': [{'qas': [_fn_answer(target) for target in targets]}]}]
    return (preds_dict, targets_dict)