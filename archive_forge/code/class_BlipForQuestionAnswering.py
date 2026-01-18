import warnings
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.functional import normalize
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_blip import BlipConfig, BlipTextConfig, BlipVisionConfig
from .modeling_blip_text import BlipTextLMHeadModel, BlipTextModel
@add_start_docstrings('\n    BLIP Model for visual question answering. The model consists of a vision encoder, a text encoder as well as a text\n    decoder. The vision encoder will encode the input image, the text encoder will encode the input question together\n    with the encoding of the image, and the text decoder will output the answer to the question.\n    ', BLIP_START_DOCSTRING)
class BlipForQuestionAnswering(BlipPreTrainedModel):
    config_class = BlipConfig
    _tied_weights_keys = ['text_decoder.cls.predictions.decoder.bias']

    def __init__(self, config: BlipConfig):
        super().__init__(config)
        self.vision_model = BlipVisionModel(config.vision_config)
        self.text_encoder = BlipTextModel(config.text_config, add_pooling_layer=False)
        self.text_decoder = BlipTextLMHeadModel(config.text_config)
        self.decoder_pad_token_id = config.text_config.pad_token_id
        self.decoder_start_token_id = config.text_config.bos_token_id
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    @add_start_docstrings_to_model_forward(BLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BlipTextVisionModelOutput, config_class=BlipVisionConfig)
    def forward(self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, labels: Optional[torch.LongTensor]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BlipTextVisionModelOutput]:
        """
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForQuestionAnswering

        >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> # training
        >>> text = "How many cats are in the picture?"
        >>> label = "2"
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> labels = processor(text=label, return_tensors="pt").input_ids

        >>> inputs["labels"] = labels
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> loss.backward()

        >>> # inference
        >>> text = "How many cats are in the picture?"
        >>> inputs = processor(images=image, text=text, return_tensors="pt")
        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```"""
        if labels is None and decoder_input_ids is None:
            raise ValueError('Either `decoder_input_ids` or `labels` should be passed when calling `forward` with `BlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`')
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        vision_outputs = self.vision_model(pixel_values=pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)
        question_embeds = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=return_dict)
        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = labels
        question_embeds = question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        answer_output = self.text_decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=question_embeds, encoder_attention_mask=attention_mask, labels=labels, return_dict=return_dict, reduction='mean')
        if labels is not None:
            decoder_loss = answer_output.loss.mean() if return_dict else answer_output[0].mean()
        else:
            decoder_loss = None
        if not return_dict:
            outputs = (decoder_loss, image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple((output for output in outputs if output is not None))
        return BlipTextVisionModelOutput(loss=decoder_loss, image_embeds=image_embeds, last_hidden_state=vision_outputs.last_hidden_state, hidden_states=vision_outputs.hidden_states, attentions=vision_outputs.attentions)

    @torch.no_grad()
    def generate(self, input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, attention_mask: Optional[torch.LongTensor]=None, **generate_kwargs) -> torch.LongTensor:
        """
        Overrides *generate* function to be able to use the model as a conditional generator

        Parameters:
            input_ids (*torch.LongTensor* of shape *(batch_size, sequence_length)*):
                The sequence used as a prompt for the generation.
            pixel_values (*torch.FloatTensor* of shape *(batch_size, num_channels, image_height, image_width)*:
                Input image to be processed
            attention_mask (*torch.LongTensor* of shape *(batch_size, sequence_length)*, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`. `1` for
                tokens that are NOT MASKED, `0` for MASKED tokens.
            **generate_kwargs:
                Additional arguments passed to the *generate* function of the decoder


        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, BlipForQuestionAnswering

        >>> model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        >>> processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are in the picture?"

        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> outputs = model.generate(**inputs)
        >>> print(processor.decode(outputs[0], skip_special_tokens=True))
        2
        ```
        """
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)
        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)
        question_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, encoder_hidden_states=image_embeds, encoder_attention_mask=image_attention_mask, return_dict=False)
        question_embeds = question_outputs[0]
        question_attention_mask = torch.ones(question_embeds.size()[:-1], dtype=torch.long).to(question_embeds.device)
        bos_ids = torch.full((question_embeds.size(0), 1), fill_value=self.decoder_start_token_id, device=question_embeds.device)
        outputs = self.text_decoder.generate(input_ids=bos_ids, eos_token_id=self.config.text_config.sep_token_id, pad_token_id=self.config.text_config.pad_token_id, encoder_hidden_states=question_embeds, encoder_attention_mask=question_attention_mask, **generate_kwargs)
        return outputs