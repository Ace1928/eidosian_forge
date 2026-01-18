import collections
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_clap import ClapAudioConfig, ClapConfig, ClapTextConfig
@add_start_docstrings(CLAP_START_DOCSTRING)
class ClapModel(ClapPreTrainedModel):
    config_class = ClapConfig

    def __init__(self, config: ClapConfig):
        super().__init__(config)
        if not isinstance(config.text_config, ClapTextConfig):
            raise ValueError(f'config.text_config is expected to be of type ClapTextConfig but is of type {type(config.text_config)}.')
        if not isinstance(config.audio_config, ClapAudioConfig):
            raise ValueError(f'config.audio_config is expected to be of type ClapAudioConfig but is of type {type(config.audio_config)}.')
        text_config = config.text_config
        audio_config = config.audio_config
        self.logit_scale_a = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.logit_scale_t = nn.Parameter(torch.tensor(math.log(config.logit_scale_init_value)))
        self.projection_dim = config.projection_dim
        self.text_model = ClapTextModel(text_config)
        self.text_projection = ClapProjectionLayer(text_config)
        self.audio_model = ClapAudioModel(audio_config)
        self.audio_projection = ClapProjectionLayer(audio_config)
        self.post_init()

    @add_start_docstrings_to_model_forward(CLAP_TEXT_INPUTS_DOCSTRING)
    def get_text_features(self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        """
        Returns:
            text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
            applying the projection layer to the pooled output of [`ClapTextModel`].

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, ClapModel

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

        >>> inputs = tokenizer(["the sound of a cat", "the sound of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = text_outputs[1] if return_dict is not None else text_outputs.pooler_output
        text_features = self.text_projection(pooled_output)
        text_features = F.normalize(text_features, dim=-1)
        return text_features

    @add_start_docstrings_to_model_forward(CLAP_AUDIO_INPUTS_DOCSTRING)
    def get_audio_features(self, input_features: Optional[torch.Tensor]=None, is_longer: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> torch.FloatTensor:
        """
        Returns:
            audio_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The audio embeddings obtained by
            applying the projection layer to the pooled output of [`ClapAudioModel`].

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor, ClapModel
        >>> import torch

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")
        >>> random_audio = torch.rand((16_000))
        >>> inputs = feature_extractor(random_audio, return_tensors="pt")
        >>> audio_features = model.get_audio_features(**inputs)
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        audio_outputs = self.audio_model(input_features=input_features, is_longer=is_longer, return_dict=return_dict)
        pooled_output = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_features = self.audio_projection(pooled_output)
        audio_features = F.normalize(audio_features, dim=-1)
        return audio_features

    @add_start_docstrings_to_model_forward(CLAP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClapOutput, config_class=ClapConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, input_features: Optional[torch.FloatTensor]=None, is_longer: Optional[torch.BoolTensor]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, return_loss: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ClapOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from datasets import load_dataset
        >>> from transformers import AutoProcessor, ClapModel

        >>> dataset = load_dataset("ashraq/esc50")
        >>> audio_sample = dataset["train"]["audio"][0]["array"]

        >>> model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        >>> processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

        >>> input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

        >>> inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

        >>> outputs = model(**inputs)
        >>> logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
        >>> probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        audio_outputs = self.audio_model(input_features=input_features, is_longer=is_longer, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        audio_embeds = audio_outputs[1] if not return_dict else audio_outputs.pooler_output
        audio_embeds = self.audio_projection(audio_embeds)
        text_embeds = text_outputs[1] if not return_dict else text_outputs.pooler_output
        text_embeds = self.text_projection(text_embeds)
        audio_embeds = audio_embeds / audio_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale_text = self.logit_scale_t.exp()
        logit_scale_audio = self.logit_scale_a.exp()
        logits_per_text = torch.matmul(text_embeds, audio_embeds.t()) * logit_scale_text
        logits_per_audio = torch.matmul(audio_embeds, text_embeds.t()) * logit_scale_audio
        loss = None
        if return_loss:
            caption_loss = contrastive_loss(logits_per_text)
            audio_loss = contrastive_loss(logits_per_audio.t())
            loss = (caption_loss + audio_loss) / 2.0
        if not return_dict:
            output = (logits_per_audio, logits_per_text, text_embeds, audio_embeds, text_outputs, audio_outputs)
            return (loss,) + output if loss is not None else output
        return ClapOutput(loss=loss, logits_per_audio=logits_per_audio, logits_per_text=logits_per_text, text_embeds=text_embeds, audio_embeds=audio_embeds, text_model_output=text_outputs, audio_model_output=audio_outputs)