import copy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_rag import RagConfig
from .retrieval_rag import RagRetriever
@add_start_docstrings_to_model_forward('\n    A RAG-token model implementation. It performs RAG-token specific marginalization in the forward pass.\n    ', RAG_START_DOCSTRING)
class RagTokenForGeneration(RagPreTrainedModel):

    def __init__(self, config: Optional[PretrainedConfig]=None, question_encoder: Optional[PreTrainedModel]=None, generator: Optional[PreTrainedModel]=None, retriever: Optional[RagRetriever]=None, **kwargs):
        assert config is not None or (question_encoder is not None and generator is not None), 'Either a configuration or an encoder and a generator has to be provided.'
        if config is None:
            config = RagConfig.from_question_encoder_generator_configs(question_encoder.config, generator.config, **kwargs)
        super().__init__(config)
        self.rag = RagModel(config=config, question_encoder=question_encoder, generator=generator, retriever=retriever)

    def set_retriever(self, retriever: RagRetriever):
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel):
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, doc_scores=None, n_docs=None, **kwargs):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'doc_scores': doc_scores, 'context_attention_mask': attention_mask, 'decoder_input_ids': decoder_input_ids, 'past_key_values': past_key_values, 'use_cache': use_cache, 'do_marginalize': True, 'n_docs': n_docs}

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Reorders cache for generation. BART-inspired but we need to take care of the extra dimension for docs"""

        def _reorder_stacked(hidden_states, new_order):
            n_docs = hidden_states.shape[0] // new_order.shape[0]
            hidden_states = hidden_states.view(-1, n_docs, *hidden_states.shape[1:])
            hidden_states = hidden_states.index_select(0, new_order)
            result = hidden_states.view(-1, *hidden_states.shape[2:])
            return result
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((_reorder_stacked(past_state, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past

    def marginalize(self, seq_logits, doc_scores, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1))
        doc_logprobs = torch.log_softmax(doc_scores, dim=1)
        log_prob_sum = seq_logprobs + doc_logprobs.unsqueeze(-1).unsqueeze(-1)
        return torch.logsumexp(log_prob_sum, dim=1)

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_retrieved: Optional[bool]=None, do_marginalize: Optional[bool]=None, reduce_loss: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, n_docs: Optional[int]=None, **kwargs) -> RetrievAugLMMarginOutput:
        """
        do_marginalize (`bool`, *optional*):
            If `True`, the logits are marginalized over all documents by making use of
            `torch.nn.functional.log_softmax`.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Legacy dictionary, which is required so that model can use *generate()* function.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagTokenForGeneration
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]
        >>> labels = targets["input_ids"]
        >>> outputs = model(input_ids=input_ids, labels=labels)

        >>> # or use retriever separately
        >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
        >>> # 1. Encode
        >>> question_hidden_states = model.question_encoder(input_ids)[0]
        >>> # 2. Retrieve
        >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
        >>> doc_scores = torch.bmm(
        ...     question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
        ... ).squeeze(1)
        >>> # 3. Forward to generator
        >>> outputs = model(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ...     decoder_input_ids=labels,
        ... )

        >>> # or directly generate
        >>> generated = model.generate(
        ...     context_input_ids=docs_dict["context_input_ids"],
        ...     context_attention_mask=docs_dict["context_attention_mask"],
        ...     doc_scores=doc_scores,
        ... )
        >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False
        outputs = self.rag(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask, doc_scores=doc_scores, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_retrieved=output_retrieved, n_docs=n_docs)
        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(outputs.logits, outputs.doc_scores, labels, reduce_loss=reduce_loss, epsilon=self.config.label_smoothing, n_docs=n_docs)
        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)
        return RetrievAugLMMarginOutput(loss=loss, logits=logits, doc_scores=outputs.doc_scores, past_key_values=outputs.past_key_values, context_input_ids=outputs.context_input_ids, context_attention_mask=outputs.context_attention_mask, retrieved_doc_embeds=outputs.retrieved_doc_embeds, retrieved_doc_ids=outputs.retrieved_doc_ids, question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state, question_enc_hidden_states=outputs.question_enc_hidden_states, question_enc_attentions=outputs.question_enc_attentions, generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state, generator_enc_hidden_states=outputs.generator_enc_hidden_states, generator_enc_attentions=outputs.generator_enc_attentions, generator_dec_hidden_states=outputs.generator_dec_hidden_states, generator_dec_attentions=outputs.generator_dec_attentions, generator_cross_attentions=outputs.generator_cross_attentions)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, n_docs: Optional[int]=None, generation_config: Optional[GenerationConfig]=None, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]]=None, logits_processor: Optional[LogitsProcessorList]=LogitsProcessorList(), stopping_criteria: Optional[StoppingCriteriaList]=StoppingCriteriaList(), **kwargs) -> torch.LongTensor:
        """
        Implements RAG token decoding.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                The sequence used as a prompt for the generation. If `input_ids` is not passed, then
                `context_input_ids` has to be provided.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            context_input_ids (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Input IDs post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model has is not initialized with a `retriever`, `context_input_ids` has to be provided to the
                forward pass. `context_input_ids` are returned by [`~RagRetriever.__call__`].
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which has the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments `inputs_ids` and the batch ID
                `batch_id`. It has to return a list with the allowed tokens for the next generation step conditioned on
                the previously generated tokens `inputs_ids` and the batch ID `batch_id`. This argument is useful for
                constrained generation conditioned on the prefix, as described in [Autoregressive Entity
                Retrieval](https://arxiv.org/abs/2010.00904).
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and a
                model's config. If a logit processor is passed that is already created with the arguments or a model's
                config an error is thrown.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                model's config. If a stopping criteria is passed that is already created with the arguments or a
                model's config an error is thrown.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model.

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            out = self.retriever(input_ids, question_hidden_states.cpu().detach().to(torch.float32).numpy(), prefix=self.generator.config.prefix, n_docs=n_docs, return_tensors='pt')
            context_input_ids, context_attention_mask, retrieved_doc_embeds = (out['context_input_ids'], out['context_attention_mask'], out['retrieved_doc_embeds'])
            retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)
            doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
        assert context_input_ids.shape[0] % n_docs == 0, f' The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}.'
        batch_size = context_input_ids.shape[0] // n_docs
        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)
        input_ids = torch.full((batch_size * generation_config.num_beams, 1), generation_config.decoder_start_token_id, dtype=torch.long, device=next(self.parameters()).device)
        input_ids_seq_length = input_ids.shape[-1]
        last_hidden_state = encoder_outputs['last_hidden_state']

        def extend_enc_output(tensor, num_beams=None):
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=generation_config.num_beams)
        encoder_outputs['last_hidden_state'] = extend_enc_output(last_hidden_state, num_beams=generation_config.num_beams)
        doc_scores = doc_scores.repeat_interleave(generation_config.num_beams, dim=0)
        model_kwargs['doc_scores'] = doc_scores
        model_kwargs['encoder_outputs'] = encoder_outputs
        model_kwargs['attention_mask'] = context_attention_mask
        model_kwargs['n_docs'] = n_docs
        pre_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=context_input_ids, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn, logits_processor=logits_processor)
        if generation_config.num_beams == 1:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1, but is {generation_config.num_return_sequences} when doing greedy search.')
            return self.greedy_search(input_ids, logits_processor=pre_processor, max_length=generation_config.max_length, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, **model_kwargs)
        elif generation_config.num_beams > 1:
            if generation_config.num_return_sequences > generation_config.num_beams:
                raise ValueError('`num_return_sequences` has to be smaller or equal to `num_beams`.')
            beam_scorer = BeamSearchScorer(batch_size=batch_size, num_beams=generation_config.num_beams, device=self.device, length_penalty=generation_config.length_penalty, do_early_stopping=generation_config.early_stopping, num_beam_hyps_to_keep=generation_config.num_return_sequences, max_length=generation_config.max_length)
            return self.beam_search(input_ids, beam_scorer, logits_processor=pre_processor, max_length=generation_config.max_length, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, **model_kwargs)
        else:
            raise ValueError(f'`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {generation_config.num_beams}')

    def get_input_embeddings(self):
        return self.rag.generator.get_input_embeddings()

    def get_output_embeddings(self):
        return self.rag.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.rag.generator.set_output_embeddings(new_embeddings)

    def shift_tokens_right(self, input_ids, start_token_id=None):
        """Shift input ids one token to the right, and pad with start_token_id"""
        if start_token_id is None:
            start_token_id = self.config.decoder_start_token_id
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = start_token_id
        return shifted_input_ids

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, n_docs=None):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1)

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return (ll.squeeze(-1), smooth_obj.squeeze(-1))
        rag_logprobs = self.marginalize(seq_logits, doc_scores, n_docs)
        target = target.unsqueeze(-1)
        assert target.dim() == rag_logprobs.dim()
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll.sum(1)
        smooth_obj = smooth_obj.sum(1)
        nll_loss = -ll
        smooth_loss = -smooth_obj
        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss