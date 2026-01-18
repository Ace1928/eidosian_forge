import copy
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
import torch
class AssistedCandidateGenerator(CandidateGenerator):
    """
    `CandidateGenerator` class to be used for assisted generation and speculative decoding. This class generates
    candidates through the use of a smaller model. Read the following blog post for more information:
    https://huggingface.co/blog/assisted-generation

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        assistant_model (`PreTrainedModel`):
            The model to be used for generating candidates. This model should be smaller than the main model.
        generation_config (`~generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        model_kwargs (`Dict`):
            The keyword arguments that will be passed to the main model, and are used as base inputs for the assistant
            model as well.
        inputs_tensor (`torch.Tensor`, *optional*):
            The model input tensor. In encoder-decoder models, this is the encoder input.
    """

    def __init__(self, input_ids: torch.LongTensor, assistant_model: 'PreTrainedModel', generation_config: 'GenerationConfig', logits_processor: 'LogitsProcessorList', model_kwargs: Dict, inputs_tensor: Optional[torch.Tensor]=None):
        device = assistant_model.device
        input_ids = input_ids.to(device)
        inputs_tensor = inputs_tensor.to(device)
        self.assistant_model = assistant_model
        self.num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens
        assistant_kwargs = {}
        for key, value in model_kwargs.items():
            if key not in ('encoder_outputs', 'assistant_encoder_outputs'):
                assistant_kwargs[key] = value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        if 'assistant_encoder_outputs' in model_kwargs:
            assistant_kwargs['encoder_outputs'] = model_kwargs['assistant_encoder_outputs']
        elif assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(inputs_tensor, assistant_model.generation_config.bos_token_id, assistant_kwargs)
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, assistant_kwargs, model_input_name)
        elif 'encoder_outputs' in model_kwargs:
            assistant_kwargs['encoder_outputs'] = model_kwargs['encoder_outputs']
        self.assistant_kwargs = assistant_kwargs
        if assistant_model.config.is_encoder_decoder:
            self.input_ids_key = 'decoder_input_ids'
            self.attention_key = 'decoder_attention_mask'
        elif 'encoder_outputs' in assistant_kwargs:
            self.input_ids_key = 'input_ids'
            self.attention_key = 'attention_mask'
            self.assistant_kwargs['attention_mask'] = self.assistant_kwargs.get('decoder_attention_mask', torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long))
        else:
            self.input_ids_key = 'input_ids'
            self.attention_key = 'attention_mask'
        eos_token_id = generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        self.eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        self.logits_processor = logits_processor
        self.generation_config = copy.deepcopy(generation_config)
        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True

    def get_candidates(self, input_ids: torch.LongTensor) -> Tuple[torch.LongTensor, Optional[torch.FloatTensor]]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        input_ids = input_ids.to(self.assistant_model.device)
        new_cur_len = input_ids.shape[-1]
        max_new_tokens = min(int(self.num_assistant_tokens), self.generation_config.max_length - new_cur_len - 1)
        if max_new_tokens == 0:
            return (input_ids, None)
        has_past_key_values = self.assistant_kwargs.get('past_key_values', None) is not None
        if has_past_key_values:
            new_cache_size = new_cur_len - 1
            self.assistant_kwargs['past_key_values'] = _crop_past_key_values(self.assistant_model, self.assistant_kwargs['past_key_values'], new_cache_size - 1)
            self.assistant_kwargs = _prepare_attention_mask(self.assistant_kwargs, new_cur_len, self.assistant_model.config.is_encoder_decoder)
            self.assistant_kwargs = _prepare_token_type_ids(self.assistant_kwargs, new_cur_len)
        assistant_generation_kwargs = {self.input_ids_key: input_ids, 'max_new_tokens': max_new_tokens, 'generation_config': self.generation_config, 'logits_processor': self.logits_processor}
        assistant_output = self.assistant_model.generate(**assistant_generation_kwargs, **self.assistant_kwargs)
        self.assistant_kwargs['past_key_values'] = assistant_output.past_key_values
        candidate_logits = torch.stack(assistant_output.scores, dim=1)
        candidate_ids = assistant_output.sequences
        return (candidate_ids, candidate_logits)

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        if self.assistant_model.generation_config.num_assistant_tokens_schedule in {'heuristic', 'heuristic_transient'}:
            if num_matches == int(self.num_assistant_tokens):
                self.num_assistant_tokens += 2.0
            else:
                self.num_assistant_tokens = max(1.0, self.num_assistant_tokens - 1.0)