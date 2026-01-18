import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
@add_start_docstrings('SpeechT5 Model with a speech encoder and a text decoder.', SPEECHT5_START_DOCSTRING)
class SpeechT5ForSpeechToText(SpeechT5PreTrainedModel):
    _tied_weights_keys = ['text_decoder_postnet.lm_head.weight']

    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        if config.vocab_size is None:
            raise ValueError(f"You are trying to instantiate {self.__class__} with a configuration that does not define the vocabulary size of the language model head. Please instantiate the model as follows: `SpeechT5ForSpeechToText.from_pretrained(..., vocab_size=vocab_size)`. or define `vocab_size` of your model's configuration.")
        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        text_decoder = SpeechT5DecoderWithTextPrenet(config)
        self.speecht5 = SpeechT5Model(config, speech_encoder, text_decoder)
        self.text_decoder_postnet = SpeechT5TextDecoderPostnet(config)
        self.post_init()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    def get_output_embeddings(self):
        return self.text_decoder_postnet.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.text_decoder_postnet.set_output_embeddings(new_embeddings)

    @add_start_docstrings_to_model_forward(SPEECHT5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.LongTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, labels: Optional[torch.LongTensor]=None) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a *.flac* or *.wav* audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (*pip install
            soundfile*). To prepare the array into `input_values`, the [`SpeechT5Processor`] should be used for padding
            and conversion into a tensor of type `torch.FloatTensor`. See [`SpeechT5Processor.__call__`] for details.

        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`SpeechT5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            SpeechT5 uses the `eos_token_id` as the starting token for `decoder_input_ids` generation. If
            `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            Label indices can be obtained using [`SpeechT5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

        Returns:

        Example:

        ```python
        >>> from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
        >>> from datasets import load_dataset

        >>> dataset = load_dataset(
        ...     "hf-internal-testing/librispeech_asr_demo", "clean", split="validation"
        ... )  # doctest: +IGNORE_RESULT
        >>> dataset = dataset.sort("id")
        >>> sampling_rate = dataset.features["audio"].sampling_rate

        >>> processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
        >>> model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")

        >>> # audio file is decoded on the fly
        >>> inputs = processor(audio=dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        >>> predicted_ids = model.generate(**inputs, max_length=100)

        >>> # transcribe speech
        >>> transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        >>> transcription[0]
        'mister quilter is the apostle of the middle classes and we are glad to welcome his gospel'
        ```

        ```python
        >>> inputs["labels"] = processor(text_target=dataset[0]["text"], return_tensors="pt").input_ids

        >>> # compute loss
        >>> loss = model(**inputs).loss
        >>> round(loss.item(), 2)
        19.68
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        outputs = self.speecht5(input_values=input_values, attention_mask=attention_mask, decoder_input_values=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, head_mask=head_mask, decoder_head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, encoder_outputs=encoder_outputs, past_key_values=past_key_values, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        logits = self.text_decoder_postnet(outputs[0])
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return Seq2SeqLMOutput(loss=loss, logits=logits, past_key_values=outputs.past_key_values, decoder_hidden_states=outputs.decoder_hidden_states, decoder_attentions=outputs.decoder_attentions, cross_attentions=outputs.cross_attentions, encoder_last_hidden_state=outputs.encoder_last_hidden_state, encoder_hidden_states=outputs.encoder_hidden_states, encoder_attentions=outputs.encoder_attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)),)
        return reordered_past