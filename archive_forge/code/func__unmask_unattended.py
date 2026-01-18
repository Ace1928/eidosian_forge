from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
@staticmethod
def _unmask_unattended(expanded_mask: torch.Tensor, attention_mask: torch.Tensor, unmasked_value: Union[bool, float]):
    """
        Attend to all tokens in masked rows from the expanded attention mask, for example the relevant first rows when
        using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        Details: https://github.com/pytorch/pytorch/issues/110213

        `expanded_mask` is [bsz, num_masks, tgt_seq_len, src_seq_len] or [bsz, tgt_seq_len, src_seq_len].
        `attention_mask` is [bsz, src_seq_len].

        The dimension num_masks of `expanded_mask` is most often 1, but it can also be the number of heads in the case of alibi attention bias.

        For example, if `attention_mask` is
        ```
        [[0, 0, 1],
         [1, 1, 1],
         [0, 1, 1]]
        ```
        and `expanded_mask` is (e.g. here left-padding case)
        ```
        [[[[0, 0, 0],
           [0, 0, 0],
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[0, 0, 0],
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        then the modified `expanded_mask` will be
        ```
        [[[[1, 1, 1],   <-- modified
           [1, 1, 1],   <-- modified
           [0, 0, 1]]],
         [[[1, 0, 0],
           [1, 1, 0],
           [1, 1, 1]]],
         [[[1, 1, 1],   <-- modified
           [0, 1, 0],
           [0, 1, 1]]]]
        ```
        """
    tmp = torch.arange(attention_mask.shape[1], 0, -1)
    indices = torch.argmax(attention_mask.cpu() * tmp, 1, keepdim=True)
    left_masked_rows = torch.where(indices > 0)[0]
    if left_masked_rows.shape[0] == 0:
        return expanded_mask
    indices = indices[left_masked_rows]
    max_len = torch.max(indices)
    range_tensor = torch.arange(max_len).unsqueeze(0)
    range_tensor = range_tensor.repeat(indices.size(0), 1)
    range_tensor[range_tensor >= indices] = 0
    if expanded_mask.dim() == 4:
        num_masks = expanded_mask.shape[1]
        if num_masks == 1:
            mask_slice = (left_masked_rows[:, None], 0, range_tensor)
        else:
            mask_slice = (left_masked_rows[:, None, None], torch.arange(num_masks)[None, :, None], range_tensor[:, None, :])
    else:
        mask_slice = (left_masked_rows[:, None], range_tensor)
    expanded_mask[mask_slice] = unmasked_value
    return expanded_mask