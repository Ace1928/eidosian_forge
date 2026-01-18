import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
@add_start_docstrings('\n    The FLAVA model for pretraining which outputs losses, embeddings, logits and transformer outputs.\n    ', FLAVA_START_DOCSTRING.format(config='FlavaConfig') + FLAVA_PRETRAINING_START_DOCSTRING_EXTRA)
class FlavaForPreTraining(FlavaPreTrainedModel):
    _tied_weights_keys = ['mmm_text_head.decoder.bias', 'mmm_image_head.decoder.bias', 'mlm_head.decoder.bias', 'mim_head.decoder.bias']

    def __init__(self, config: FlavaConfig, image_codebook: Optional[nn.Module]=None):
        super().__init__(config)
        self.flava = FlavaModel(config)
        self.image_codebook = image_codebook
        if self.image_codebook is None and config.init_codebook:
            self.image_codebook = FlavaImageCodebook(config.image_codebook_config)
        self.mim_head = FlavaMaskedPredictionHead(config.image_config)
        self.mlm_head = FlavaMaskedPredictionHead(config.text_config)
        self.itm_head = FlavaITMHead(config)
        self.mmm_image_head = FlavaMaskedPredictionHead(config.image_config)
        self.mmm_text_head = FlavaMaskedPredictionHead(config.text_config)
        self.global_contrastive_head = FlavaGlobalContrastiveHead(config)
        self.image_vocab_size = config.image_config.vocab_size
        self.text_vocab_size = config.text_config.vocab_size
        self.mlm_weight = config.mlm_weight
        self.mim_weight = config.mim_weight
        self.global_contrastive_weight = config.global_contrastive_weight
        self.ce_ignore_index = config.ce_ignore_index
        self.itm_weight = config.itm_weight
        self.mmm_image_weight = config.mmm_image_weight
        self.mmm_text_weight = config.mmm_text_weight
        self.skip_unmasked_multimodal_encoder = config.skip_unmasked_multimodal_encoder
        self.post_init()

    def _resize_to_2d(self, x: torch.Tensor):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return x

    @add_start_docstrings_to_model_forward(FLAVA_PRETRAINING_INPUTS_DOCSTRING.format('batch_size, text_seq_len', 'batch_size, image_num_patches'))
    @replace_return_docstrings(output_type=FlavaForPreTrainingOutput, config_class=FlavaConfig)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, input_ids_masked: Optional[torch.LongTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, codebook_pixel_values: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=None, bool_masked_pos: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, image_attention_mask: Optional[torch.Tensor]=None, skip_unmasked_multimodal_encoder: bool=None, mlm_labels: Optional[torch.Tensor]=None, mim_labels: Optional[torch.Tensor]=None, itm_labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: bool=True, return_dict: Optional[bool]=None, return_loss: Optional[bool]=None) -> Union[Tuple[torch.Tensor], FlavaForPreTrainingOutput]:
        """
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import FlavaForPreTraining, AutoProcessor

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> model = FlavaForPreTraining.from_pretrained("facebook/flava-full")
        >>> processor = AutoProcessor.from_pretrained("facebook/flava-full")

        >>> text = ["a photo of a cat"]

        >>> inputs = processor(
        ...     images=[image],
        ...     text=text,
        ...     return_masks=True,
        ...     return_codebook_pixels=True,
        ...     padding=True,
        ...     max_length=77,
        ...     return_tensors="pt",
        ... )


        >>> output = model(**inputs)
        ```

        Return:

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        return_loss = return_loss if return_loss is not None else self.config.return_loss
        skip_unmasked_multimodal_encoder = skip_unmasked_multimodal_encoder if skip_unmasked_multimodal_encoder is not None else self.skip_unmasked_multimodal_encoder
        if input_ids_masked is None and input_ids is not None:
            logger.warning("`input_ids_masked` isn't passed which means MLM loss won't be calculated correctlySetting it to `input_ids` so that model can work. Please pass it if this is unintentional. This is usually OKAY if you are doing inference on unmasked text...")
            input_ids_masked = input_ids
        flava_output = self.flava(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, image_attention_mask=image_attention_mask, skip_multimodal_encoder=skip_unmasked_multimodal_encoder, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        flava_masked_output = self.flava(input_ids=input_ids_masked, pixel_values=pixel_values, attention_mask=attention_mask, token_type_ids=token_type_ids, image_attention_mask=image_attention_mask, bool_masked_pos=bool_masked_pos, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        pos_mask = None
        image_embeddings = flava_output.image_embeddings
        text_embeddings = flava_output.text_embeddings
        image_masked_embeddings = flava_masked_output.image_embeddings
        text_masked_embeddings = flava_masked_output.text_embeddings
        multimodal_masked_embeddings = flava_masked_output.multimodal_embeddings
        total_loss = mim_loss = mlm_loss = mmm_text_loss = mmm_image_loss = gc_loss = itm_loss = None
        mim_logits = mlm_logits = mmm_text_logits = mmm_image_logits = None
        itm_logits = logits_per_image = logits_per_text = None
        if image_masked_embeddings is not None or multimodal_masked_embeddings is not None:
            if mim_labels is None and return_loss:
                if self.image_codebook is None:
                    raise RuntimeError('`return_loss` is set to True but the image codebook is not initialized and no `mim_labels`  have been passed. Reinstantiate the model with `init_codebook` set to True or pass in your custom `mim_labels`')
                if codebook_pixel_values is None:
                    raise ValueError('`codebook_pixel_value` are required to generate `mim_labels` if loss is expected. Call `AutoProcessor` with `return_codebook_pixels` set to True')
                mim_labels = self.image_codebook.get_codebook_indices(codebook_pixel_values)
        if self.mim_weight > 0 and image_masked_embeddings is not None and (multimodal_masked_embeddings is None):
            sequence_for_image = image_masked_embeddings
            if mim_labels is not None:
                mim_labels = self._resize_to_2d(mim_labels)
                bool_masked_pos = self._resize_to_2d(bool_masked_pos)
                mim_labels[bool_masked_pos.ne(True)] = self.ce_ignore_index
                sequence_for_image = sequence_for_image[:, -mim_labels.size(1):, :]
                masked_tokens = mim_labels.ne(self.ce_ignore_index)
                mim_labels_filtered = mim_labels[masked_tokens]
                sequence_for_image = sequence_for_image[masked_tokens, :]
                mim_logits = self.mim_head(sequence_for_image)
                if return_loss:
                    mim_loss = nn.functional.cross_entropy(mim_logits.view(-1, self.image_vocab_size), mim_labels_filtered.view(-1))
                    mim_loss *= self.mim_weight
            else:
                mim_logits = self.mim_head(sequence_for_image)
        if self.mlm_weight > 0 and text_masked_embeddings is not None and (multimodal_masked_embeddings is None):
            sequence_for_text = text_masked_embeddings
            if mlm_labels is not None:
                mlm_labels = self._resize_to_2d(mlm_labels)
                sequence_for_text = sequence_for_text[:, -mlm_labels.size(1):, :]
                masked_tokens = mlm_labels.ne(self.ce_ignore_index)
                mlm_labels_filtered = mlm_labels[masked_tokens]
                sequence_for_text = sequence_for_text[masked_tokens, :]
                mlm_logits = self.mlm_head(sequence_for_text)
                if return_loss:
                    mlm_loss = nn.functional.cross_entropy(mlm_logits.view(-1, self.text_vocab_size), mlm_labels_filtered.view(-1))
                    mlm_loss *= self.mlm_weight
            else:
                mlm_logits = self.mlm_head(sequence_for_text)
        if self.itm_weight > 0 and multimodal_masked_embeddings is not None:
            itm_logits = self.itm_head(multimodal_masked_embeddings)
            if itm_labels is not None:
                pos_pairs = itm_labels.ne(0)
                pos_mask = torch.where(pos_pairs.any(), pos_pairs, pos_pairs.new([True]))
                if return_loss:
                    itm_loss = nn.functional.cross_entropy(itm_logits, itm_labels)
                    itm_loss *= self.itm_weight
                if multimodal_masked_embeddings is not None:
                    multimodal_masked_embeddings = multimodal_masked_embeddings[pos_mask]
                if mlm_labels is not None:
                    mlm_labels = mlm_labels[pos_mask]
                if mim_labels is not None:
                    mim_labels = mim_labels[pos_mask]
                    bool_masked_pos = bool_masked_pos[pos_mask]
        if multimodal_masked_embeddings is not None and self.mmm_image_weight > 0:
            sequence_for_image = multimodal_masked_embeddings
            end_index = image_masked_embeddings.size(1) - 1
            sequence_for_image = sequence_for_image[:, 2:2 + end_index, :]
            if mim_labels is not None:
                mim_labels = self._resize_to_2d(mim_labels)
                bool_masked_pos = self._resize_to_2d(bool_masked_pos)
                mim_labels[bool_masked_pos.ne(True)] = self.ce_ignore_index
                masked_tokens = mim_labels.ne(self.ce_ignore_index)
                mim_labels_filtered = mim_labels[masked_tokens]
                sequence_for_image = sequence_for_image[masked_tokens, :]
                mmm_image_logits = self.mmm_image_head(sequence_for_image)
                if return_loss:
                    mmm_image_loss = nn.functional.cross_entropy(mmm_image_logits.view(-1, self.image_vocab_size), mim_labels_filtered.view(-1))
                    mmm_image_loss *= self.mmm_image_weight
            else:
                mmm_image_logits = self.mmm_image_head(sequence_for_image)
        if multimodal_masked_embeddings is not None and self.mmm_text_weight > 0:
            sequence_for_text = multimodal_masked_embeddings
            sequence_for_text = sequence_for_text[:, -text_masked_embeddings.size(1):, :]
            if mlm_labels is not None:
                mlm_labels = self._resize_to_2d(mlm_labels)
                masked_tokens = mlm_labels.ne(self.ce_ignore_index)
                mlm_labels_filtered = mlm_labels[masked_tokens]
                sequence_for_text = sequence_for_text[masked_tokens, :]
                mmm_text_logits = self.mmm_text_head(sequence_for_text)
                if return_loss:
                    mmm_text_loss = nn.functional.cross_entropy(mmm_text_logits.view(-1, self.text_vocab_size), mlm_labels_filtered.view(-1))
                    mmm_text_loss *= self.mmm_text_weight
            else:
                mmm_text_logits = self.mmm_text_head(sequence_for_text)
        if image_embeddings is not None and text_embeddings is not None and (self.global_contrastive_weight > 0):
            text_embedding = self.flava.text_projection(text_embeddings[:, 0, :])
            text_embedding = nn.functional.normalize(text_embedding, dim=-1)
            image_embedding = self.flava.image_projection(image_embeddings[:, 0, :])
            image_embedding = nn.functional.normalize(image_embedding, dim=-1)
            self.flava.logit_scale.data.clamp_(LOGIT_SCALE_CLAMP_MIN, LOGIT_SCALE_CLAMP_MAX)
            logits_per_image, logits_per_text, gc_labels = self.global_contrastive_head(image_embedding, text_embedding, self.flava.logit_scale)
            if pos_mask is not None:
                logits_per_image = logits_per_image[pos_mask]
                logits_per_text = logits_per_text[pos_mask]
                gc_labels = gc_labels[pos_mask]
            if return_loss:
                gc_loss_image = nn.functional.cross_entropy(logits_per_image, gc_labels)
                gc_loss_text = nn.functional.cross_entropy(logits_per_text, gc_labels)
                gc_loss = (gc_loss_image + gc_loss_text) / 2
                gc_loss *= self.global_contrastive_weight
        flava_losses = FlavaLosses(mim=mim_loss, mlm=mlm_loss, itm=itm_loss, global_contrastive=gc_loss, mmm_image=mmm_image_loss, mmm_text=mmm_text_loss)
        if return_loss and (not flava_losses.all_none()):
            total_loss = sum((loss if loss is not None else 0 for loss in flava_losses.values()))
        if not return_dict:
            output = (image_embeddings, flava_output.image_output.to_tuple() if flava_output.image_output is not None else None, text_embeddings, flava_output.text_output.to_tuple() if flava_output.text_output is not None else None, flava_output.multimodal_embeddings, flava_output.multimodal_output.to_tuple() if flava_output.multimodal_output is not None else None, image_masked_embeddings, flava_masked_output.image_output.to_tuple() if flava_masked_output.image_output is not None else None, text_masked_embeddings, flava_masked_output.text_output.to_tuple() if flava_masked_output.text_output is not None else None, multimodal_masked_embeddings, flava_masked_output.multimodal_output.to_tuple() if flava_masked_output.multimodal_output is not None else None, mim_logits, mlm_logits, itm_logits, logits_per_image, logits_per_image, mmm_image_logits, mmm_text_logits)
            if return_loss and (not flava_losses.all_none()):
                output = (total_loss, flava_losses) + output
            return tuple((x for x in output if x is None))
        return FlavaForPreTrainingOutput(loss=total_loss, loss_info=flava_losses, image_embeddings=image_embeddings, image_output=flava_output.image_output, text_embeddings=text_embeddings, text_output=flava_output.text_output, multimodal_embeddings=flava_output.multimodal_embeddings, multimodal_output=flava_output.multimodal_output, image_masked_embeddings=image_masked_embeddings, image_masked_output=flava_masked_output.image_output, text_masked_embeddings=text_masked_embeddings, text_masked_output=flava_masked_output.text_output, multimodal_masked_embeddings=multimodal_masked_embeddings, multimodal_masked_output=flava_masked_output.multimodal_output, mim_logits=mim_logits, mlm_logits=mlm_logits, itm_logits=itm_logits, contrastive_logits_per_image=logits_per_image, contrastive_logits_per_text=logits_per_text, mmm_image_logits=mmm_image_logits, mmm_text_logits=mmm_text_logits)