import dataclasses
import functools
import inspect
import math
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import transformers
from packaging import version
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet
from transformers.utils import is_torch_available
from ...configuration_utils import _transformers_version
from ...utils import logging
class SpeechT5ModelPatcher(ModelPatcher):

    def __enter__(self):
        self.patch_ops()
        self._model.speecht5.decoder.prenet.forward = types.MethodType(patched_speecht5_prenet_forward, self._model.speecht5.decoder.prenet)
        setattr(self._model, self.orig_forward_name, self.patched_forward)

    def __exit__(self, exc_type, exc_value, traceback):
        self.restore_ops()
        setattr(self._model, self.orig_forward_name, self.orig_forward)
        self._model.speecht5.decoder.prenet.forward = types.MethodType(self.original_speecht5_prenet_forward, self._model.speecht5.decoder.prenet)

    def __init__(self, config: 'OnnxConfig', model: Union['PreTrainedModel', 'TFPreTrainedModel'], model_kwargs: Dict[str, Any]):
        super().__init__(config, model, model_kwargs)
        self.original_speecht5_prenet_forward = model.speecht5.decoder.prenet.forward
        model.vocoder = model_kwargs['vocoder_model'].eval()

        def patched_forward(input_ids=None, speaker_embeddings=None, encoder_outputs=None, past_key_values=None, output_sequence=None, spectrogram=None, encoder_attention_mask=None):
            use_cache = self.real_config.use_past and self.real_config.variant == 'with-past'
            if self.real_config._behavior == 'encoder':
                encoder_attention_mask = torch.ones_like(input_ids)
                encoder_out = model.speecht5.encoder(input_values=input_ids, attention_mask=encoder_attention_mask, return_dict=True)
                if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
                    encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(encoder_out[0].shape[1], encoder_attention_mask)
                result = {'encoder_outputs': encoder_out.last_hidden_state, 'encoder_attention_mask': encoder_attention_mask}
            elif self.real_config._behavior == 'decoder':
                encoder_hidden_states = encoder_outputs[0]
                decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence, speaker_embeddings)
                decoder_out = model.speecht5.decoder.wrapped_decoder(hidden_states=decoder_hidden_states[:, -1:], attention_mask=None, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, past_key_values=past_key_values, use_cache=use_cache, output_attentions=False, return_dict=True)
                last_decoder_output = decoder_out.last_hidden_state[0, -1]
                past_key_values = decoder_out.past_key_values
                spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output)
                spectrum = spectrum.view(model.config.reduction_factor, model.config.num_mel_bins)
                output_sequence = torch.cat((output_sequence, spectrum[-1].view(1, 1, model.config.num_mel_bins)), dim=1)
                prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))
                result = {'output_sequence_out': output_sequence, 'spectrum': spectrum, 'prob': prob, 'past_key_values': past_key_values}
            elif self.real_config.is_postnet_and_vocoder:
                spectrogram = spectrogram.unsqueeze(0)
                spectrogram = model.speech_decoder_postnet.postnet(spectrogram)
                spectrogram = spectrogram.squeeze(0)
                waveform = model.vocoder(spectrogram)
                result = {'waveform': waveform}
            else:
                raise ValueError('Should not happen')
            filterd_outputs = {}
            for name, value in result.items():
                if name != 'past_key_values':
                    filterd_outputs[name] = value
                elif self.real_config._behavior == 'decoder' and (self.real_config.is_merged or not self.real_config.use_past_in_inputs):
                    filterd_outputs[name] = value
                elif self.real_config._behavior == 'decoder' and self.real_config.use_past_in_inputs:
                    filterd_outputs[name] = tuple([v[:2] for v in value])
            return filterd_outputs
        self.patched_forward = patched_forward