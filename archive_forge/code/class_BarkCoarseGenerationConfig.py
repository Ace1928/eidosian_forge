import copy
from typing import Dict
from ...generation.configuration_utils import GenerationConfig
from ...utils import logging
class BarkCoarseGenerationConfig(GenerationConfig):
    model_type = 'coarse_acoustics'

    def __init__(self, renormalize_logits=True, output_scores=False, return_dict_in_generate=False, output_hidden_states=False, output_attentions=False, temperature=1.0, do_sample=False, coarse_semantic_pad_token=12048, coarse_rate_hz=75, n_coarse_codebooks=2, coarse_infer_token=12050, max_coarse_input_length=256, max_coarse_history: int=630, sliding_window_len: int=60, **kwargs):
        """Class that holds a generation configuration for [`BarkCoarseModel`].

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            renormalize_logits (`bool`, *optional*, defaults to `True`):
                Whether to renormalize the logits after applying all the logits processors or warpers (including the
                custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the
                score logits are normalized but some logit processors or warpers break the normalization.
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
            coarse_semantic_pad_token (`int`, *optional*, defaults to 12_048):
                Coarse semantic pad token.
            coarse_rate_hz (`int`, *optional*, defaults to 75):
                Coarse rate in Hertz.
            n_coarse_codebooks (`int`, *optional*, defaults to 2):
                Number of coarse codebooks.
            coarse_infer_token (`int`, *optional*, defaults to 12_050):
                Coarse infer token.
            max_coarse_input_length (`int`, *optional*, defaults to 256):
                Max length of input coarse vector.
            max_coarse_history (`int`, *optional*, defaults to 630):
                Max length of the output of the coarse acoustics model used in the fine generation step.
            sliding_window_len (`int`, *optional*, defaults to 60):
                The coarse generation step uses a sliding window to generate raw audio.
        """
        super().__init__(temperature=temperature, do_sample=do_sample, renormalize_logits=renormalize_logits, output_scores=output_scores, return_dict_in_generate=return_dict_in_generate, output_hidden_states=output_hidden_states, output_attentions=output_attentions, **kwargs)
        self.coarse_semantic_pad_token = coarse_semantic_pad_token
        self.coarse_rate_hz = coarse_rate_hz
        self.n_coarse_codebooks = n_coarse_codebooks
        self.coarse_infer_token = coarse_infer_token
        self.max_coarse_input_length = max_coarse_input_length
        self.max_coarse_history = max_coarse_history
        self.sliding_window_len = sliding_window_len