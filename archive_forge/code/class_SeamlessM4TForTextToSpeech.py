import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_seamless_m4t import SeamlessM4TConfig
@add_start_docstrings('The text-to-speech SeamlessM4T Model transformer which can be used for T2ST.', SEAMLESS_M4T_START_DOCSTRING)
class SeamlessM4TForTextToSpeech(SeamlessM4TPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['speech_encoder']
    main_input_name = 'input_ids'
    _tied_weights_keys = ['lm_head.weight', 'text_encoder.embed_tokens.weight', 'text_decoder.embed_tokens.weight']

    def __init__(self, config: SeamlessM4TConfig):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.text_encoder = SeamlessM4TEncoder(config, self.shared)
        self.text_decoder = SeamlessM4TDecoder(config, self.shared)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.t2u_model = SeamlessM4TTextToUnitForConditionalGeneration(config)
        self.vocoder = SeamlessM4TCodeHifiGan(config)

    def get_encoder(self):
        return self.text_encoder

    def get_decoder(self):
        return self.text_decoder

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.text_decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.text_encoder.embed_tokens = value
        self.text_decoder.embed_tokens = value
        self.shared = value

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.text_encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.text_decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.lm_head, self.shared)

    @add_start_docstrings_to_model_forward(M4T_TEXT_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Seq2SeqLMOutput, Tuple[torch.FloatTensor]]:
        if labels is not None:
            if use_cache:
                logger.warning('The `use_cache` argument is changed to `False` since `labels` is provided.')
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            logger.warning("This is the same forward method as `SeamlessM4TForTextToText`.It doesn't use the text-to-unit model `SeamlessM4TTextToUnitForConditionalGeneration`.If you want to generate speech, use the `.generate` method.")
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        encoder_attention_mask = attention_mask
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_outputs[0], encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, inputs_embeds=decoder_inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        lm_logits = self.lm_head(decoder_outputs[0])
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            labels = labels.to(lm_logits.device)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            outputs = decoder_outputs + encoder_outputs
            output = (lm_logits,) + outputs[1:]
            return (masked_lm_loss,) + output if masked_lm_loss is not None else output
        return Seq2SeqLMOutput(loss=masked_lm_loss, logits=lm_logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    @torch.no_grad()
    def generate(self, input_ids: Optional[torch.Tensor]=None, return_intermediate_token_ids: Optional[bool]=None, tgt_lang: Optional[str]=None, spkr_id: Optional[int]=0, **kwargs) -> Union[torch.Tensor, SeamlessM4TGenerationOutput]:
        """
        Generates translated audio waveforms.

        <Tip>

        This method successively calls the `.generate` function of two different sub-models. You can specify keyword
        arguments at two different levels: general arguments that will be passed to both models, or prefixed arguments
        that will be passed to one of them.

        For example, calling `.generate(input_ids, num_beams=4, speech_do_sample=True)` will successively perform
        beam-search decoding on the text model, and multinomial beam-search sampling on the speech model.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

                Indices can be obtained using [`SeamlessM4TTokenizer`] or [`SeamlessM4TProcessor`]. See
                [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            return_intermediate_token_ids (`bool`, *optional*):
                If `True`, also returns the intermediate generated text and unit tokens. Set to `True` if you also want
                to get translated text alongside the audio.
            tgt_lang (`str`, *optional*):
                The language to use as target language for translation.
            spkr_id (`int`, *optional*, defaults to 0):
                The id of the speaker used for speech synthesis. Must be lower than `config.vocoder_num_spkrs`.
            kwargs (*optional*):
                Remaining dictionary of keyword arguments that will be passed to [`GenerationMixin.generate`]. Keyword
                arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model,
                    except for `decoder_input_ids` which will only be passed through the text components.
                    - With a *text_* or *speech_* prefix, they will be input for the `generate` method of the
                    text model and speech model respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for one generation but not for the
                    other.


        Returns:
            `Union[SeamlessM4TGenerationOutput, Tuple[Tensor]]`:
            - If `return_intermediate_token_ids`, returns [`SeamlessM4TGenerationOutput`].
            - If not `return_intermediate_token_ids`, returns a tuple composed of waveforms of shape `(batch_size,
              sequence_length)`and and `waveform_lengths` which gives the length of each sample.
        """
        batch_size = len(input_ids) if input_ids is not None else len(kwargs.get('inputs_embeds'))
        if tgt_lang is None:
            raise ValueError('You must specify a `tgt_lang` to generate translated speech.')
        else:
            tgt_lang = tgt_lang.replace('__', '')
            for key in ['text_decoder_lang_to_code_id', 't2u_lang_code_to_id', 'vocoder_lang_code_to_id']:
                lang_code_to_id = getattr(self.generation_config, key, None)
                if lang_code_to_id is None:
                    raise ValueError(f"This model generation config doesn't have a `{key}` key which maps the target language\n                        to the right token id. Make sure to load the right generation config.")
                elif tgt_lang not in lang_code_to_id:
                    raise ValueError(f'`tgt_lang={tgt_lang}` is not supported by this model.\n                    Please specify a `tgt_lang` in {','.join(lang_code_to_id.keys())}. Note that SeamlessM4T supports\n                    more languages for text translation than for speech synthesis.')
        kwargs_text, kwargs_speech = format_speech_generation_kwargs(kwargs)
        kwargs_text['output_hidden_states'] = True
        kwargs_text['return_dict_in_generate'] = True
        kwargs_text['output_scores'] = True
        text_decoder_input_ids = kwargs_text.get('decoder_input_ids')
        text_tgt_lang_id = self.generation_config.text_decoder_lang_to_code_id.get(tgt_lang)
        text_decoder_input_ids = torch.tensor([[text_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_text['decoder_input_ids'] = text_decoder_input_ids
        text_generation_output = super().generate(input_ids, **kwargs_text)
        sequences = text_generation_output.sequences
        num_return_sequences = len(sequences) // batch_size
        attention_mask = kwargs_speech.get('attention_mask', kwargs_text.get('attention_mask', None))
        encoder_hidden_states = text_generation_output.encoder_hidden_states[-1]
        if num_return_sequences > 1:
            idx_most_probable_sequences_per_batch = text_generation_output.sequences_scores.view(batch_size, -1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch.argmax(-1)
            idx_most_probable_sequences_per_batch = idx_most_probable_sequences_per_batch + torch.arange(batch_size).to(self.device) * num_return_sequences
            sequences = sequences[idx_most_probable_sequences_per_batch]
        t2u_input_embeds = self.text_decoder(input_ids=sequences, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask).last_hidden_state
        pad_token_id = self.generation_config.pad_token_id
        seq_lens = (sequences != pad_token_id).int().sum(1)
        t2u_model_attention_mask = _compute_new_attention_mask(t2u_input_embeds, seq_lens)
        kwargs_speech['attention_mask'] = t2u_model_attention_mask
        t2u_decoder_input_ids = kwargs_speech.get('decoder_input_ids')
        t2u_tgt_lang_id = self.generation_config.t2u_lang_code_to_id.get(tgt_lang)
        t2u_decoder_input_ids = torch.tensor([[self.config.t2u_eos_token_id, t2u_tgt_lang_id]] * batch_size).to(self.device)
        kwargs_speech['decoder_input_ids'] = t2u_decoder_input_ids
        unit_ids = self.t2u_model.generate(inputs_embeds=t2u_input_embeds, **kwargs_speech)
        output_unit_ids = unit_ids.detach().clone()
        unit_ids = unit_ids[:, kwargs_speech['decoder_input_ids'].shape[1]:]
        unit_ids[unit_ids == self.config.t2u_eos_token_id] = self.config.t2u_pad_token_id
        unit_ids = torch.where(unit_ids == self.config.t2u_pad_token_id, unit_ids, unit_ids - self.config.vocoder_offset)
        vocoder_tgt_lang_id = self.generation_config.vocoder_lang_code_to_id.get(tgt_lang)
        vocoder_tgt_lang_id = torch.tensor([[vocoder_tgt_lang_id]] * len(unit_ids)).to(self.device)
        spkr_id = torch.tensor([[spkr_id]] * len(unit_ids)).to(self.device)
        waveform, waveform_lengths = self.vocoder(input_ids=unit_ids, spkr_id=spkr_id, lang_id=vocoder_tgt_lang_id)
        if return_intermediate_token_ids:
            return SeamlessM4TGenerationOutput(waveform=waveform, waveform_lengths=waveform_lengths, sequences=sequences, unit_sequences=output_unit_ids)
        return (waveform, waveform_lengths)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx) for past_state in layer_past[:2])) + layer_past[2:],)
        return reordered_past