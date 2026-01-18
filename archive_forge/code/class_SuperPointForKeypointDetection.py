from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
@add_start_docstrings('SuperPoint model outputting keypoints and descriptors.', SUPERPOINT_START_DOCSTRING)
class SuperPointForKeypointDetection(SuperPointPreTrainedModel):
    """
    SuperPoint model. It consists of a SuperPointEncoder, a SuperPointInterestPointDecoder and a
    SuperPointDescriptorDecoder. SuperPoint was proposed in `SuperPoint: Self-Supervised Interest Point Detection and
    Description <https://arxiv.org/abs/1712.07629>`__ by Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. It
    is a fully convolutional neural network that extracts keypoints and descriptors from an image. It is trained in a
    self-supervised manner, using a combination of a photometric loss and a loss based on the homographic adaptation of
    keypoints. It is made of a convolutional encoder and two decoders: one for keypoints and one for descriptors.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__(config)
        self.config = config
        self.encoder = SuperPointEncoder(config)
        self.keypoint_decoder = SuperPointInterestPointDecoder(config)
        self.descriptor_decoder = SuperPointDescriptorDecoder(config)
        self.post_init()

    @add_start_docstrings_to_model_forward(SUPERPOINT_INPUTS_DOCSTRING)
    def forward(self, pixel_values: torch.FloatTensor, labels: Optional[torch.LongTensor]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple, SuperPointKeypointDescriptionOutput]:
        """
        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SuperPointForKeypointDetection
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
        >>> model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")

        >>> inputs = processor(image, return_tensors="pt")
        >>> outputs = model(**inputs)
        ```"""
        loss = None
        if labels is not None:
            raise ValueError('SuperPoint does not support training for now.')
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        pixel_values = self.extract_one_channel_pixel_values(pixel_values)
        batch_size = pixel_values.shape[0]
        encoder_outputs = self.encoder(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)
        last_hidden_state = encoder_outputs[0]
        list_keypoints_scores = [self.keypoint_decoder(last_hidden_state[None, ...]) for last_hidden_state in last_hidden_state]
        list_keypoints = [keypoints_scores[0] for keypoints_scores in list_keypoints_scores]
        list_scores = [keypoints_scores[1] for keypoints_scores in list_keypoints_scores]
        list_descriptors = [self.descriptor_decoder(last_hidden_state[None, ...], keypoints[None, ...]) for last_hidden_state, keypoints in zip(last_hidden_state, list_keypoints)]
        maximum_num_keypoints = max((keypoints.shape[0] for keypoints in list_keypoints))
        keypoints = torch.zeros((batch_size, maximum_num_keypoints, 2), device=pixel_values.device)
        scores = torch.zeros((batch_size, maximum_num_keypoints), device=pixel_values.device)
        descriptors = torch.zeros((batch_size, maximum_num_keypoints, self.config.descriptor_decoder_dim), device=pixel_values.device)
        mask = torch.zeros((batch_size, maximum_num_keypoints), device=pixel_values.device, dtype=torch.int)
        for i, (_keypoints, _scores, _descriptors) in enumerate(zip(list_keypoints, list_scores, list_descriptors)):
            keypoints[i, :_keypoints.shape[0]] = _keypoints
            scores[i, :_scores.shape[0]] = _scores
            descriptors[i, :_descriptors.shape[0]] = _descriptors
            mask[i, :_scores.shape[0]] = 1
        hidden_states = encoder_outputs[1] if output_hidden_states else None
        if not return_dict:
            return tuple((v for v in [loss, keypoints, scores, descriptors, mask, hidden_states] if v is not None))
        return SuperPointKeypointDescriptionOutput(loss=loss, keypoints=keypoints, scores=scores, descriptors=descriptors, mask=mask, hidden_states=hidden_states)