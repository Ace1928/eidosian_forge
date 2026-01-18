import csv
import urllib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.functional.text.helper_embedding_metric import (
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.checks import _SKIP_SLOW_DOCTEST, _try_proceed_with_timeout
from torchmetrics.utilities.imports import _TQDM_AVAILABLE, _TRANSFORMERS_GREATER_EQUAL_4_4
def _get_embeddings_and_idf_scale(dataloader: DataLoader, target_len: int, model: Module, device: Optional[Union[str, torch.device]]=None, num_layers: Optional[int]=None, all_layers: bool=False, idf: bool=False, verbose: bool=False, user_forward_fn: Optional[Callable[[Module, Dict[str, Tensor]], Tensor]]=None) -> Tuple[Tensor, Tensor]:
    """Calculate sentence embeddings and the inverse-document-frequency scaling factor.

    Args:
        dataloader: dataloader instance.
        target_len: A length of the longest sequence in the data. Used for padding the model output.
        model: BERT model.
        device: A device to be used for calculation.
        num_layers: The layer of representation to use.
        all_layers: An indication whether representation from all model layers should be used for BERTScore.
        idf: An Indication whether normalization using inverse document frequencies should be used.
        verbose: An indication of whether a progress bar to be displayed during the embeddings' calculation.
        user_forward_fn:
            A user's own forward function used in a combination with ``user_model``. This function must
            take ``user_model`` and a python dictionary of containing ``"input_ids"`` and ``"attention_mask"``
            represented by :class:`~torch.Tensor` as an input and return the model's output represented by the single
            :class:`~torch.Tensor`.

    Return:
        A tuple of :class:`~torch.Tensor`s containing the model's embeddings and the normalized tokens IDF.
        When ``idf = False``, tokens IDF is not calculated, and a matrix of mean weights is returned instead.
        For a single sentence, ``mean_weight = 1/seq_len``, where ``seq_len`` is a sum over the corresponding
        ``attention_mask``.

    Raises:
        ValueError:
            If ``all_layers = True`` and a model, which is not from the ``transformers`` package, is used.

    """
    embeddings_list: List[Tensor] = []
    idf_scale_list: List[Tensor] = []
    for batch in _get_progress_bar(dataloader, verbose):
        with torch.no_grad():
            batch = _input_data_collator(batch, device)
            if not all_layers:
                if not user_forward_fn:
                    out = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)
                    out = out.hidden_states[num_layers if num_layers is not None else -1]
                else:
                    out = user_forward_fn(model, batch)
                    _check_shape_of_model_output(out, batch['input_ids'])
                out = out.unsqueeze(1)
            else:
                if user_forward_fn:
                    raise ValueError('The option `all_layers=True` can be used only with default `transformers` models.')
                out = model(batch['input_ids'], batch['attention_mask'], output_hidden_states=True)
                out = torch.cat([o.unsqueeze(1) for o in out.hidden_states], dim=1)
        out /= out.norm(dim=-1).unsqueeze(-1)
        out, attention_mask = _output_data_collator(out, batch['attention_mask'], target_len)
        processed_attention_mask = _process_attention_mask_for_special_tokens(attention_mask)
        out = torch.einsum('blsd, bs -> blsd', out, processed_attention_mask)
        embeddings_list.append(out.cpu())
        input_ids_idf = batch['input_ids_idf'] * processed_attention_mask if idf else processed_attention_mask.type(out.dtype)
        input_ids_idf /= input_ids_idf.sum(-1, keepdim=True)
        idf_scale_list.append(input_ids_idf.cpu())
    embeddings = torch.cat(embeddings_list)
    idf_scale = torch.cat(idf_scale_list)
    return (embeddings, idf_scale)