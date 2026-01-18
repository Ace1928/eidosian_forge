import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch import nn
from ..cache_utils import Cache, DynamicCache, StaticCache
from ..integrations.deepspeed import is_deepspeed_zero3_enabled
from ..modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput
from ..models.auto import (
from ..utils import ModelOutput, is_accelerate_available, is_torchdynamo_compiling, logging
from .beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from .beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from .candidate_generator import (
from .configuration_utils import GenerationConfig, GenerationMode
from .logits_process import (
from .stopping_criteria import (
def _assisted_decoding(self, input_ids: torch.LongTensor, candidate_generator: Optional['CandidateGenerator']=None, do_sample: bool=False, logits_processor: Optional[LogitsProcessorList]=None, logits_warper: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[Union[int, List[int]]]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_scores: Optional[bool]=None, output_logits: Optional[bool]=None, return_dict_in_generate: Optional[bool]=None, synced_gpus: bool=False, streamer: Optional['BaseStreamer']=None, **model_kwargs) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    """
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** or
        **sample** (depending on `do_sample`), assisted by candidate sequences. Assisted generation is an example of a
        candidate decoding strategy. Can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text
        models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin._assisted_decoding`] directly. Use
        generate() instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            candidate_generator (`CandidateGenerator`, *optional*):
                A derived instance of [`CandidateGenerator`] that defines how candidate sequences are generated. For
                more information, the documentation of [`CandidateGenerator`] should be read.
            do_sample (`bool`, *optional*, defaults to `False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            output_logits (`bool`, *optional*, defaults to `False`):
                Whether or not to return the raw prediction logit scores. See `logits` under returned tensors for
                more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> from transformers.generation import AssistedCandidateGenerator

        >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        >>> assistant_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id
        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])
        >>> candidate_generator = AssistedCandidateGenerator(
        ...     input_ids=input_ids,
        ...     assistant_model=assistant_model,
        ...     generation_config=model.generation_config,
        ...     logits_processor=logits_processor,
        ...     model_kwargs={},
        ... )
        >>> outputs = model._assisted_decoding(
        ...     input_ids,
        ...     candidate_generator=candidate_generator,
        ...     logits_processor=logits_processor,
        ...     stopping_criteria=stopping_criteria,
        ... )
        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    if eos_token_id is not None:
        logger.warning_once('`eos_token_id` is deprecated in this function and will be removed in v4.41, use `stopping_criteria=StoppingCriteriaList([EosTokenCriteria(eos_token_id=eos_token_id)])` instead. Otherwise make sure to set `model.generation_config.eos_token_id`', FutureWarning)
        stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    else:
        eos_token_id = [criteria.eos_token_id.tolist() for criteria in stopping_criteria if hasattr(criteria, 'eos_token_id')]
        eos_token_id = eos_token_id[0] if eos_token_id else None
        if eos_token_id is None and self.generation_config.eos_token_id is not None:
            eos_token_id = self.generation_config.eos_token_id
            stopping_criteria.append(EosTokenCriteria(eos_token_id=eos_token_id))
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
    output_attentions = output_attentions if output_attentions is not None else self.generation_config.output_attentions
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    return_dict_in_generate = return_dict_in_generate if return_dict_in_generate is not None else self.generation_config.return_dict_in_generate
    scores = () if return_dict_in_generate and output_scores else None
    raw_logits = () if return_dict_in_generate and output_logits else None
    decoder_attentions = () if return_dict_in_generate and output_attentions else None
    cross_attentions = () if return_dict_in_generate and output_attentions else None
    decoder_hidden_states = () if return_dict_in_generate and output_hidden_states else None
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs['encoder_outputs'].get('attentions') if output_attentions else None
        encoder_hidden_states = model_kwargs['encoder_outputs'].get('hidden_states') if output_hidden_states else None
    batch_size, cur_len = input_ids.shape
    if 'inputs_embeds' in model_kwargs:
        cur_len = model_kwargs['inputs_embeds'].shape[1]
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs['cache_position'] = torch.arange(cur_len, device=input_ids.device)
    this_peer_finished = False
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        cur_len = input_ids.shape[-1]
        candidate_input_ids, candidate_logits = candidate_generator.get_candidates(input_ids)
        candidate_input_ids = candidate_input_ids.to(self.device)
        if candidate_logits is not None:
            candidate_logits = candidate_logits.to(self.device)
        candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]
        is_done_candidate = stopping_criteria(candidate_input_ids, None)
        model_kwargs = _prepare_attention_mask(model_kwargs, candidate_input_ids.shape[1], self.config.is_encoder_decoder)
        model_kwargs = _prepare_token_type_ids(model_kwargs, candidate_input_ids.shape[1])
        if 'cache_position' in model_kwargs:
            model_kwargs['cache_position'] = torch.cat((model_kwargs['cache_position'], torch.arange(cur_len, cur_len + candidate_length, device=input_ids.device, dtype=torch.long)), dim=0)
        model_inputs = self.prepare_inputs_for_generation(candidate_input_ids, **model_kwargs)
        if 'num_logits_to_keep' in model_inputs:
            model_inputs['num_logits_to_keep'] = candidate_length + 1
        outputs = self(**model_inputs, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        new_logits = outputs.logits[:, -candidate_length - 1:]
        next_token_logits = new_logits.clone()
        if len(logits_processor) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_processor(candidate_input_ids[:, :cur_len + i], new_logits[:, i, :])
        if len(logits_warper) > 0:
            for i in range(candidate_length + 1):
                new_logits[:, i, :] = logits_warper(candidate_input_ids[:, :cur_len + i], new_logits[:, i, :])
        if do_sample and candidate_logits is not None:
            valid_tokens, n_matches = _speculative_sampling(candidate_input_ids, candidate_logits, candidate_length, new_logits, is_done_candidate)
        else:
            if do_sample:
                probs = new_logits.softmax(dim=-1)
                selected_tokens = torch.multinomial(probs[0, :, :], num_samples=1).squeeze(1)[None, :]
            else:
                selected_tokens = new_logits.argmax(dim=-1)
            candidate_new_tokens = candidate_input_ids[:, cur_len:]
            n_matches = ((~(candidate_new_tokens == selected_tokens[:, :-1])).cumsum(dim=-1) < 1).sum()
            if is_done_candidate and n_matches == candidate_length:
                n_matches -= 1
            valid_tokens = selected_tokens[:, :n_matches + 1]
        input_ids = torch.cat((input_ids, valid_tokens), dim=-1)
        if streamer is not None:
            streamer.put(valid_tokens.cpu())
        new_cur_len = input_ids.shape[-1]
        new_cache_size = new_cur_len - 1
        outputs.past_key_values = _crop_past_key_values(self, outputs.past_key_values, new_cache_size)
        candidate_generator.update_candidate_strategy(input_ids, new_logits, n_matches)
        if synced_gpus and this_peer_finished:
            continue
        if return_dict_in_generate:
            if output_scores:
                scores += tuple((new_logits[:, i, :] for i in range(n_matches + 1)))
            if output_logits:
                raw_logits += (next_token_logits,)
            if 'past_key_values' not in model_kwargs:
                added_len = new_cur_len
            else:
                added_len = n_matches + 1
            if output_attentions:
                if self.config.is_encoder_decoder:
                    cross_attentions = _split_model_outputs(cross_attentions, outputs.cross_attentions, cur_len, added_len)
                    decoder_attentions = _split_model_outputs(decoder_attentions, outputs.decoder_attentions, cur_len, added_len, is_decoder_attention=True)
                else:
                    decoder_attentions = _split_model_outputs(decoder_attentions, outputs.attentions, cur_len, added_len, is_decoder_attention=True)
            if output_hidden_states:
                if self.config.is_encoder_decoder:
                    decoder_hidden_states = _split_model_outputs(decoder_hidden_states, outputs.decoder_hidden_states, cur_len, added_len)
                else:
                    decoder_hidden_states = _split_model_outputs(decoder_hidden_states, outputs.hidden_states, cur_len, added_len)
        model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
    if streamer is not None:
        streamer.end()
    if hasattr(candidate_generator, 'assistant_model') and candidate_generator.assistant_model.generation_config.num_assistant_tokens_schedule == 'heuristic':
        candidate_generator.assistant_model.generation_config.num_assistant_tokens = candidate_generator.num_assistant_tokens
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(sequences=input_ids, scores=scores, logits=raw_logits, encoder_attentions=encoder_attentions, encoder_hidden_states=encoder_hidden_states, decoder_attentions=decoder_attentions, cross_attentions=cross_attentions, decoder_hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
        else:
            return GenerateDecoderOnlyOutput(sequences=input_ids, scores=scores, logits=raw_logits, attentions=decoder_attentions, hidden_states=decoder_hidden_states, past_key_values=model_kwargs.get('past_key_values'))
    else:
        return input_ids