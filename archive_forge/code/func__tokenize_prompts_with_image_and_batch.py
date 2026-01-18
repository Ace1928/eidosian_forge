import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ...utils import TensorType, is_torch_available, logging, requires_backends
def _tokenize_prompts_with_image_and_batch(tokenizer, prompts: List[List[str]], scale_factors: Optional[List[List['torch.Tensor']]], max_tokens_to_generate: int, max_position_embeddings: int, add_BOS: bool, add_beginning_of_answer_token: bool) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Given a set of prompts and number of tokens to generate:
    - tokenize prompts
    - set the sequence length to be the max of length of prompts plus the number of tokens we would like to generate
    - pad all the sequences to this length so we can convert them into a 3D tensor.
    """
    if scale_factors is not None:
        transformed_prompt_tokens = []
        for prompt_seq, scale_factor_seq in zip(prompts, scale_factors):
            transformed_prompt_tokens.append([_transform_coordinates_and_tokenize(prompt, scale_factor.item(), tokenizer) for prompt, scale_factor in zip(prompt_seq, scale_factor_seq)])
    else:
        transformed_prompt_tokens = [[tokenizer.tokenize(prompt) for prompt in prompt_seq] for prompt_seq in prompts]
    prompts_tokens = transformed_prompt_tokens
    if add_BOS:
        bos_token = tokenizer.vocab['<s>']
    else:
        bos_token = tokenizer.vocab['|ENDOFTEXT|']
    prompts_tokens = [[[bos_token] + x for x in prompt_seq] for prompt_seq in prompts_tokens]
    if add_beginning_of_answer_token:
        boa = tokenizer.vocab[BEGINNING_OF_ANSWER_STRING]
        for token_seq in prompts_tokens:
            token_seq[-1].append(boa)
    prompts_length = [[len(x) for x in prompts_tokens_seq] for prompts_tokens_seq in prompts_tokens]
    max_prompt_len: int = np.max(prompts_length)
    samples_length = min(max_prompt_len + max_tokens_to_generate, max_position_embeddings)
    if max_prompt_len + max_tokens_to_generate > max_position_embeddings:
        logger.warning(f'Max subsequence prompt length of {max_prompt_len} + max tokens to generate {max_tokens_to_generate}', f'exceeds context length of {max_position_embeddings}. Will generate as many tokens as possible.')
    for prompt_tokens_seq, prompts_length_seq in zip(prompts_tokens, prompts_length):
        for prompt_tokens, prompt_length in zip(prompt_tokens_seq, prompts_length_seq):
            if len(prompt_tokens) > samples_length:
                raise ValueError('Length of subsequence prompt exceeds sequence length.')
            padding_size = samples_length - prompt_length
            prompt_tokens.extend([tokenizer.vocab['|ENDOFTEXT|']] * padding_size)
    prompts_tokens_tensor = torch.tensor(prompts_tokens, dtype=torch.int64)
    prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.int64)
    return (prompts_tokens_tensor, prompts_length_tensor)