import copy
from typing import Dict
from ...generation.configuration_utils import GenerationConfig
from ...utils import logging
class BarkSemanticGenerationConfig(GenerationConfig):
    model_type = 'semantic'

    def __init__(self, eos_token_id=10000, renormalize_logits=True, max_new_tokens=768, output_scores=False, return_dict_in_generate=False, output_hidden_states=False, output_attentions=False, temperature=1.0, do_sample=False, text_encoding_offset=10048, text_pad_token=129595, semantic_infer_token=129599, semantic_vocab_size=10000, max_input_semantic_length=256, semantic_rate_hz=49.9, min_eos_p=None, **kwargs):
        """Class that holds a generation configuration for [`BarkSemanticModel`].

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            eos_token_id (`int`, *optional*, defaults to 10_000):
                The id of the *end-of-sequence* token.
            renormalize_logits (`bool`, *optional*, defaults to `True`):
                Whether to renormalize the logits after applying all the logits processors or warpers (including the
                custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
                score logits are normalized but some logit processors or warpers break the normalization.
            max_new_tokens (`int`, *optional*, defaults to 768):
                The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            temperature (`float`, *optional*, defaults to 1.0):
                The value used to modulate the next token probabilities.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            text_encoding_offset (`int`, *optional*, defaults to 10_048):
                Text encoding offset.
            text_pad_token (`int`, *optional*, defaults to 129_595):
                Text pad token.
            semantic_infer_token (`int`, *optional*, defaults to 129_599):
                Semantic infer token.
            semantic_vocab_size (`int`, *optional*, defaults to 10_000):
                Semantic vocab size.
            max_input_semantic_length (`int`, *optional*, defaults to 256):
                Max length of semantic input vector.
            semantic_rate_hz (`float`, *optional*, defaults to 49.9):
                Semantic rate in Hertz.
            min_eos_p (`float`, *optional*):
                Minimum threshold of the probability of the EOS token for it to be sampled. This is an early stopping
                strategy to mitigate potential unwanted generations at the end of a prompt. The original implementation
                suggests a default value of 0.2.
        """
        super().__init__(temperature=temperature, do_sample=do_sample, eos_token_id=eos_token_id, renormalize_logits=renormalize_logits, max_new_tokens=max_new_tokens, output_scores=output_scores, return_dict_in_generate=return_dict_in_generate, output_hidden_states=output_hidden_states, output_attentions=output_attentions, **kwargs)
        self.text_encoding_offset = text_encoding_offset
        self.text_pad_token = text_pad_token
        self.semantic_pad_token = eos_token_id
        self.semantic_infer_token = semantic_infer_token
        self.semantic_vocab_size = semantic_vocab_size
        self.max_input_semantic_length = max_input_semantic_length
        self.semantic_rate_hz = semantic_rate_hz
        self.min_eos_p = min_eos_p