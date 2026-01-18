import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
@add_start_docstrings('The bare LayoutLMv2 Model transformer outputting raw hidden-states without any specific head on top.', LAYOUTLMV2_START_DOCSTRING)
class LayoutLMv2Model(LayoutLMv2PreTrainedModel):

    def __init__(self, config):
        requires_backends(self, 'detectron2')
        super().__init__(config)
        self.config = config
        self.has_visual_segment_embedding = config.has_visual_segment_embedding
        self.embeddings = LayoutLMv2Embeddings(config)
        self.visual = LayoutLMv2VisualBackbone(config)
        self.visual_proj = nn.Linear(config.image_feature_pool_shape[-1], config.hidden_size)
        if self.has_visual_segment_embedding:
            self.visual_segment_embedding = nn.Parameter(nn.Embedding(1, config.hidden_size).weight[0])
        self.visual_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.visual_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.encoder = LayoutLMv2Encoder(config)
        self.pooler = LayoutLMv2Pooler(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _calc_text_embeddings(self, input_ids, bbox, position_ids, token_type_ids, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if inputs_embeds is None:
            inputs_embeds = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + spatial_position_embeddings + token_type_embeddings
        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings

    def _calc_img_embeddings(self, image, bbox, position_ids):
        visual_embeddings = self.visual_proj(self.visual(image))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        spatial_position_embeddings = self.embeddings._calc_spatial_position_embeddings(bbox)
        embeddings = visual_embeddings + position_embeddings + spatial_position_embeddings
        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings

    def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
        visual_bbox_x = torch.div(torch.arange(0, 1000 * (image_feature_pool_shape[1] + 1), 1000, device=device, dtype=bbox.dtype), self.config.image_feature_pool_shape[1], rounding_mode='floor')
        visual_bbox_y = torch.div(torch.arange(0, 1000 * (self.config.image_feature_pool_shape[0] + 1), 1000, device=device, dtype=bbox.dtype), self.config.image_feature_pool_shape[0], rounding_mode='floor')
        visual_bbox = torch.stack([visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1), visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1), visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1), visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1)], dim=-1).view(-1, bbox.size(-1))
        visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)
        return visual_bbox

    def _get_input_shape(self, input_ids=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            return input_ids.size()
        elif inputs_embeds is not None:
            return inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')

    @add_start_docstrings_to_model_forward(LAYOUTLMV2_INPUTS_DOCSTRING.format('(batch_size, sequence_length)'))
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, bbox: Optional[torch.LongTensor]=None, image: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, token_type_ids: Optional[torch.LongTensor]=None, position_ids: Optional[torch.LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None, inputs_embeds: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Return:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, LayoutLMv2Model, set_seed
        >>> from PIL import Image
        >>> import torch
        >>> from datasets import load_dataset

        >>> set_seed(88)

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv2-base-uncased")
        >>> model = LayoutLMv2Model.from_pretrained("microsoft/layoutlmv2-base-uncased")


        >>> dataset = load_dataset("hf-internal-testing/fixtures_docvqa")
        >>> image_path = dataset["test"][0]["file"]
        >>> image = Image.open(image_path).convert("RGB")

        >>> encoding = processor(image, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> last_hidden_states = outputs.last_hidden_state

        >>> last_hidden_states.shape
        torch.Size([1, 342, 768])
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = self._get_input_shape(input_ids, inputs_embeds)
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        visual_shape = list(input_shape)
        visual_shape[1] = self.config.image_feature_pool_shape[0] * self.config.image_feature_pool_shape[1]
        visual_shape = torch.Size(visual_shape)
        final_shape = list(self._get_input_shape(input_ids, inputs_embeds))
        final_shape[1] += visual_shape[1]
        final_shape = torch.Size(final_shape)
        visual_bbox = self._calc_visual_bbox(self.config.image_feature_pool_shape, bbox, device, final_shape)
        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        visual_attention_mask = torch.ones(visual_shape, device=device)
        final_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if position_ids is None:
            seq_length = input_shape[1]
            position_ids = self.embeddings.position_ids[:, :seq_length]
            position_ids = position_ids.expand(input_shape)
        visual_position_ids = torch.arange(0, visual_shape[1], dtype=torch.long, device=device).repeat(input_shape[0], 1)
        final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
        if bbox is None:
            bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)
        text_layout_emb = self._calc_text_embeddings(input_ids=input_ids, bbox=bbox, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        visual_emb = self._calc_img_embeddings(image=image, bbox=visual_bbox, position_ids=visual_position_ids)
        final_emb = torch.cat([text_layout_emb, visual_emb], dim=1)
        extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers
        encoder_outputs = self.encoder(final_emb, extended_attention_mask, bbox=final_bbox, position_ids=final_position_ids, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)