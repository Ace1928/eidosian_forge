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
@add_start_docstrings_to_model_forward('\n    A RAG-sequence model implementation. It performs RAG-sequence specific marginalization in the forward pass.\n    ', RAG_START_DOCSTRING)
class RagSequenceForGeneration(RagPreTrainedModel):

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

    @add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=RetrievAugLMMarginOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.Tensor]]]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, output_retrieved: Optional[bool]=None, exclude_bos_score: Optional[bool]=None, reduce_loss: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, n_docs: Optional[int]=None, **kwargs) -> RetrievAugLMMarginOutput:
        """
        exclude_bos_score (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the score of the BOS token is disregarded when computing
            the loss.
        reduce_loss (`bool`, *optional*):
            Only relevant if `labels` is passed. If `True`, the NLL loss is reduced using the `torch.Tensor.sum`
            operation.
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
             Legacy dictionary, which is required so that model can use *generate()* function.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
        >>> retriever = RagRetriever.from_pretrained(
        ...     "facebook/rag-sequence-nq", index_name="exact", use_dummy_dataset=True
        ... )
        >>> # initialize with RagRetriever to do everything in one forward call
        >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

        >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
        >>> targets = tokenizer(text_target="In Paris, there are 10 million people.", return_tensors="pt")
        >>> input_ids = inputs["input_ids"]
        >>> labels = targets["input_ids"]
        >>> outputs = model(input_ids=input_ids, labels=labels)

        >>> # or use retriever separately
        >>> model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", use_dummy_dataset=True)
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
        ```"""
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        exclude_bos_score = exclude_bos_score if exclude_bos_score is not None else self.config.exclude_bos_score
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False
        outputs = self.rag(input_ids=input_ids, attention_mask=attention_mask, encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, context_input_ids=context_input_ids, context_attention_mask=context_attention_mask, doc_scores=doc_scores, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, output_retrieved=output_retrieved, n_docs=n_docs)
        loss = None
        if labels is not None:
            loss = self.get_nll(outputs.logits, outputs.doc_scores, decoder_input_ids, reduce_loss=reduce_loss, epsilon=self.config.label_smoothing, exclude_bos_score=exclude_bos_score, n_docs=n_docs)
        return RetrievAugLMMarginOutput(loss=loss, logits=outputs.logits, doc_scores=outputs.doc_scores, past_key_values=outputs.past_key_values, context_input_ids=outputs.context_input_ids, context_attention_mask=outputs.context_attention_mask, retrieved_doc_embeds=outputs.retrieved_doc_embeds, retrieved_doc_ids=outputs.retrieved_doc_ids, question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state, question_enc_hidden_states=outputs.question_enc_hidden_states, question_enc_attentions=outputs.question_enc_attentions, generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state, generator_enc_hidden_states=outputs.generator_enc_hidden_states, generator_enc_attentions=outputs.generator_enc_attentions, generator_dec_hidden_states=outputs.generator_dec_hidden_states, generator_dec_attentions=outputs.generator_dec_attentions, generator_cross_attentions=outputs.generator_cross_attentions)

    @property
    def retriever(self):
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, context_input_ids: Optional[torch.LongTensor]=None, context_attention_mask: Optional[torch.LongTensor]=None, doc_scores: Optional[torch.FloatTensor]=None, do_deduplication: Optional[bool]=None, num_return_sequences: Optional[int]=None, num_beams: Optional[int]=None, n_docs: Optional[int]=None, **model_kwargs) -> torch.LongTensor:
        """
        Implements RAG sequence "thorough" decoding. Read the [`~generation.GenerationMixin.generate`]` documentation
        for more information on how to set other generate input parameters.

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
                Input IDs post-processed from the retrieved documents and the question encoder input_ids by the
                retriever.
            context_attention_mask (`torch.LongTensor` of shape `(batch_size * config.n_docs, config.max_combined_length)`, *optional*, returned when *output_retrieved=True*):
                Attention mask post-processed from the retrieved documents and the question encoder `input_ids` by the
                retriever.

                If the model is not initialized with a `retriever` or `input_ids` is not given, `context_input_ids` and
                `context_attention_mask` have to be provided to the forward pass. They are returned by
                [`~RagRetriever.__call__`].
            doc_scores (`torch.FloatTensor` of shape `(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see `retrieved_doc_embeds`) and
                `question_encoder_last_hidden_state`.

                If the model is not initialized with a `retriever` or `input_ids` is not given, `doc_scores` has to be
                provided to the forward pass. `doc_scores` are returned by [`~RagRetriever.__call__`].
            do_deduplication (`bool`, *optional*):
                Whether or not to deduplicate the generations from different context documents for a given input. Has
                to be set to `False` if used while training with distributed backend.
            num_return_sequences(`int`, *optional*, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the `generator`'s `[`~generation.GenerationMixin.generate`]` function,
                where we set `num_return_sequences` to `num_beams`.
            num_beams (`int`, *optional*, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            n_docs (`int`, *optional*, defaults to `config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs will be passed to [`~generation.GenerationMixin.generate`].

        Return:
            `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence length) is either equal to `max_length` or shorter if all batches
            finished early due to the `eos_token_id`.
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_deduplication = do_deduplication if do_deduplication is not None else self.config.do_deduplication
        num_doc_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        assert input_ids is not None or context_input_ids is not None, ' At least one of input_ids or context_input_ids must be given'
        if self.retriever is not None and context_input_ids is None:
            question_hidden_states = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
            context_input_ids = self.retriever(input_ids, question_hidden_states.cpu().detach().to(torch.float32).numpy(), prefix=self.generator.config.prefix, n_docs=n_docs, return_tensors='pt')['context_input_ids']
            context_input_ids = context_input_ids.to(input_ids)
        hypos = []
        model_kwargs['num_beams'] = num_beams
        model_kwargs['num_return_sequences'] = num_beams
        model_kwargs['attention_mask'] = None
        batch_size = input_ids.shape[0] if input_ids is not None else context_input_ids.shape[0] // n_docs
        for index in range(batch_size):
            generator_input_ids = context_input_ids[index * n_docs:(index + 1) * n_docs]
            output_sequences = self.generator.generate(generator_input_ids, **model_kwargs)
            if do_deduplication:
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))
            num_candidates = output_sequences.shape[0]
            if input_ids is not None:
                new_input_ids = input_ids[index:index + 1].repeat(num_candidates, 1)
                outputs = self(new_input_ids, labels=output_sequences, exclude_bos_score=True)
            else:
                assert context_attention_mask is not None, 'Make sure that `context_attention_mask` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                assert doc_scores is not None, 'Make sure that `doc_scores` are passed, if no `input_ids` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function.'
                individual_input_ids = generator_input_ids.repeat(num_candidates, 1)
                individual_attention_mask = context_attention_mask[index * n_docs:(index + 1) * n_docs]
                individual_attention_mask = individual_attention_mask.repeat(num_candidates, 1)
                individual_doc_scores = doc_scores[index:index + 1, :]
                individual_doc_scores = individual_doc_scores.repeat(num_candidates, 1)
                outputs = self(context_input_ids=individual_input_ids, context_attention_mask=individual_attention_mask, doc_scores=individual_doc_scores, labels=output_sequences, exclude_bos_score=True)
            top_cand_inds = (-outputs['loss']).topk(num_doc_return_sequences)[1]
            hypos.append(output_sequences[top_cand_inds])
        return self._cat_and_pad(hypos, pad_token_id=self.config.generator.pad_token_id)

    def get_nll(self, seq_logits, doc_scores, target, reduce_loss=False, epsilon=0.0, exclude_bos_score=False, n_docs=None):
        target = torch.cat([target[:, 1:], target.new(target.shape[0], 1).fill_(self.config.generator.pad_token_id)], 1)
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        bos_token_id = self.config.bos_token_id or self.config.generator.bos_token_id
        use_bos = bos_token_id is not None and target[:, 0].eq(bos_token_id).all()

        def _mask_pads(ll, smooth_obj):
            pad_mask = target.eq(self.config.generator.pad_token_id)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
                smooth_obj.masked_fill_(pad_mask, 0.0)
            return (ll.squeeze(-1), smooth_obj.squeeze(-1))
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(seq_logits.shape[0] // n_docs, n_docs, -1, seq_logits.size(-1))
        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=1).unsqueeze(-1).unsqueeze(-1)
        first_token_scores = seq_logprobs[:, :, :1, :]
        second_token_scores = seq_logprobs[:, :, 1:2, :]
        remainder = seq_logprobs[:, :, 2:, :]
        rag_logprobs = torch.cat([first_token_scores, second_token_scores + doc_logprobs, remainder], dim=2)
        target = target.unsqueeze(1).unsqueeze(-1).repeat(1, n_docs, 1, 1)
        assert target.dim() == rag_logprobs.dim()
        ll = rag_logprobs.gather(dim=-1, index=target)
        smooth_obj = rag_logprobs.sum(dim=-1, keepdim=True)
        ll, smooth_obj = _mask_pads(ll, smooth_obj)
        ll = ll[:, :, 1:].sum(2) if exclude_bos_score and use_bos else ll.sum(2)
        smooth_obj = smooth_obj.sum(2)
        ll = ll.logsumexp(1)
        smooth_obj = smooth_obj.logsumexp(1)
        nll_loss = -ll
        smooth_loss = -smooth_obj
        if reduce_loss:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        eps_i = epsilon / rag_logprobs.size(-1)
        loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
        return loss

    @staticmethod
    def _cat_and_pad(tensors, pad_token_id):
        output = tensors[0].new(sum([t.shape[0] for t in tensors]), max([t.shape[1] for t in tensors])).fill_(pad_token_id)
        ind = 0
        for t in tensors:
            output[ind:ind + t.shape[0], :t.shape[1]] = t
            ind += t.shape[0]
        return output