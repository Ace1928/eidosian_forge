import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageSuperResolutionOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_swin2sr import Swin2SRConfig
@add_start_docstrings('\n    Swin2SR Model transformer with an upsampler head on top for image super resolution and restoration.\n    ', SWIN2SR_START_DOCSTRING)
class Swin2SRForImageSuperResolution(Swin2SRPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.swin2sr = Swin2SRModel(config)
        self.upsampler = config.upsampler
        self.upscale = config.upscale
        num_features = 64
        if self.upsampler == 'pixelshuffle':
            self.upsample = PixelShuffleUpsampler(config, num_features)
        elif self.upsampler == 'pixelshuffle_aux':
            self.upsample = PixelShuffleAuxUpsampler(config, num_features)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(config.upscale, config.embed_dim, config.num_channels_out)
        elif self.upsampler == 'nearest+conv':
            self.upsample = NearestConvUpsampler(config, num_features)
        else:
            self.final_convolution = nn.Conv2d(config.embed_dim, config.num_channels_out, 3, 1, 1)
        self.post_init()

    @add_start_docstrings_to_model_forward(SWIN2SR_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ImageSuperResolutionOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, pixel_values: Optional[torch.FloatTensor]=None, head_mask: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, ImageSuperResolutionOutput]:
        """
        Returns:

        Example:
         ```python
         >>> import torch
         >>> import numpy as np
         >>> from PIL import Image
         >>> import requests

         >>> from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

         >>> processor = AutoImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
         >>> model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

         >>> url = "https://huggingface.co/spaces/jjourney1125/swin2sr/resolve/main/samples/butterfly.jpg"
         >>> image = Image.open(requests.get(url, stream=True).raw)
         >>> # prepare image for the model
         >>> inputs = processor(image, return_tensors="pt")

         >>> # forward pass
         >>> with torch.no_grad():
         ...     outputs = model(**inputs)

         >>> output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
         >>> output = np.moveaxis(output, source=0, destination=-1)
         >>> output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
         >>> # you can visualize `output` with `Image.fromarray`
         ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        height, width = pixel_values.shape[2:]
        if self.config.upsampler == 'pixelshuffle_aux':
            bicubic = nn.functional.interpolate(pixel_values, size=(height * self.upscale, width * self.upscale), mode='bicubic', align_corners=False)
        outputs = self.swin2sr(pixel_values, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        if self.upsampler in ['pixelshuffle', 'pixelshuffledirect', 'nearest+conv']:
            reconstruction = self.upsample(sequence_output)
        elif self.upsampler == 'pixelshuffle_aux':
            reconstruction, aux = self.upsample(sequence_output, bicubic, height, width)
            aux = aux / self.swin2sr.img_range + self.swin2sr.mean
        else:
            reconstruction = pixel_values + self.final_convolution(sequence_output)
        reconstruction = reconstruction / self.swin2sr.img_range + self.swin2sr.mean
        reconstruction = reconstruction[:, :, :height * self.upscale, :width * self.upscale]
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not supported at the moment')
        if not return_dict:
            output = (reconstruction,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return ImageSuperResolutionOutput(loss=loss, reconstruction=reconstruction, hidden_states=outputs.hidden_states, attentions=outputs.attentions)