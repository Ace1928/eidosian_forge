import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import torch
def _prepare_token_type_ids(model_kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """Expands or crops the model's token_type_ids for decoding purposes, to the defined length"""
    if 'token_type_ids' not in model_kwargs or model_kwargs['token_type_ids'] is None:
        return model_kwargs
    token_type_ids = model_kwargs['token_type_ids']
    final_token_type = token_type_ids[:, -1].unsqueeze(-1)
    type_length_diff = new_length - token_type_ids.shape[1]
    if type_length_diff < 0:
        token_type_ids = token_type_ids[:, :type_length_diff]
    elif type_length_diff > 0:
        token_type_copies = final_token_type.repeat(1, type_length_diff)
        model_kwargs['token_type_ids'] = torch.cat([model_kwargs['token_type_ids'], token_type_copies], dim=-1)
    return model_kwargs