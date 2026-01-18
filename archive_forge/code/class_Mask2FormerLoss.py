import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import is_accelerate_available, logging
from ...utils.backbone_utils import load_backbone
from .configuration_mask2former import Mask2FormerConfig
class Mask2FormerLoss(nn.Module):

    def __init__(self, config: Mask2FormerConfig, weight_dict: Dict[str, float]):
        """
        The Mask2Former Loss. The loss is computed very similar to DETR. The process happens in two steps: 1) we
        compute hungarian assignment between ground truth masks and the outputs of the model 2) we supervise each pair
        of matched ground-truth / prediction (supervise class and mask)

        Args:
            config (`Mask2FormerConfig`):
                The configuration for Mask2Former model also containing loss calculation specific parameters.
            weight_dict (`Dict[str, float]`):
                A dictionary of weights to be applied to the different losses.
        """
        super().__init__()
        requires_backends(self, ['scipy'])
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict
        self.eos_coef = config.no_object_weight
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio
        self.matcher = Mask2FormerHungarianMatcher(cost_class=1.0, cost_dice=config.dice_weight, cost_mask=config.mask_weight, num_points=self.num_points)

    def _max_by_axis(self, sizes: List[List[int]]) -> List[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(self, tensors: List[Tensor]) -> Tuple[Tensor, Tensor]:
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape
        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for tensor, padded_tensor, padding_mask in zip(tensors, padded_tensors, padding_masks):
            padded_tensor[:tensor.shape[0], :tensor.shape[1], :tensor.shape[2]].copy_(tensor)
            padding_mask[:tensor.shape[1], :tensor.shape[2]] = False
        return (padded_tensors, padding_masks)

    def loss_labels(self, class_queries_logits: Tensor, class_labels: List[Tensor], indices: Tuple[np.array]) -> Dict[str, Tensor]:
        """Compute the losses related to the labels using cross entropy.

        Args:
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `batch_size, num_queries, num_labels`
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.

        Returns:
            `Dict[str, Tensor]`: A dict of `torch.Tensor` containing the following key:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
        """
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat([target[j] for target, (_, j) in zip(class_labels, indices)])
        target_classes = torch.full((batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device)
        target_classes[idx] = target_classes_o
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)
        losses = {'loss_cross_entropy': loss_ce}
        return losses

    def loss_masks(self, masks_queries_logits: torch.Tensor, mask_labels: List[torch.Tensor], indices: Tuple[np.array], num_masks: int) -> Dict[str, torch.Tensor]:
        """Compute the losses related to the masks using sigmoid_cross_entropy_loss and dice loss.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            indices (`Tuple[np.array])`:
                The indices computed by the Hungarian matcher.
            num_masks (`int)`:
                The number of masks, used for normalization.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing two keys:
            - **loss_mask** -- The loss computed using sigmoid cross entropy loss on the predicted and ground truth.
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth,
              masks.
        """
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]
        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]
        with torch.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(pred_masks, lambda logits: self.calculate_uncertainty(logits), self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            point_labels = sample_point(target_masks, point_coordinates, align_corners=False).squeeze(1)
        point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)
        losses = {'loss_mask': sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks), 'loss_dice': dice_loss(point_logits, point_labels, num_masks)}
        del pred_masks
        del target_masks
        return losses

    def _get_predictions_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        predictions_indices = torch.cat([src for src, _ in indices])
        return (batch_indices, predictions_indices)

    def _get_targets_permutation_indices(self, indices):
        batch_indices = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        target_indices = torch.cat([tgt for _, tgt in indices])
        return (batch_indices, target_indices)

    def calculate_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        In Mask2Former paper, uncertainty is estimated as L1 distance between 0.0 and the logit prediction in 'logits'
        for the foreground class in `classes`.

        Args:
            logits (`torch.Tensor`):
            A tensor of shape (R, 1, ...) for class-specific or class-agnostic, where R is the total number of predicted masks in all images and C is:
            the number of foreground classes. The values are logits.

        Returns:
            scores (`torch.Tensor`): A tensor of shape (R, 1, ...) that contains uncertainty scores with the most
            uncertain locations having the highest uncertainty score.
        """
        uncertainty_scores = -torch.abs(logits)
        return uncertainty_scores

    def sample_points_using_uncertainty(self, logits: torch.Tensor, uncertainty_function, num_points: int, oversample_ratio: int, importance_sample_ratio: float) -> torch.Tensor:
        """
        This function is meant for sampling points in [0, 1] * [0, 1] coordinate space based on their uncertainty. The
        uncertainty is calculated for each point using the passed `uncertainty function` that takes points logit
        prediction as input.

        Args:
            logits (`float`):
                Logit predictions for P points.
            uncertainty_function:
                A function that takes logit predictions for P points and returns their uncertainties.
            num_points (`int`):
                The number of points P to sample.
            oversample_ratio (`int`):
                Oversampling parameter.
            importance_sample_ratio (`float`):
                Ratio of points that are sampled via importance sampling.

        Returns:
            point_coordinates (`torch.Tensor`):
                Coordinates for P sampled points.
        """
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)
        point_coordinates = torch.rand(num_boxes, num_points_sampled, 2, device=logits.device)
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_points_sampled * torch.arange(num_boxes, dtype=torch.long, device=logits.device)
        idx += shift[:, None]
        point_coordinates = point_coordinates.view(-1, 2)[idx.view(-1), :].view(num_boxes, num_uncertain_points, 2)
        if num_random_points > 0:
            point_coordinates = torch.cat([point_coordinates, torch.rand(num_boxes, num_random_points, 2, device=logits.device)], dim=1)
        return point_coordinates

    def forward(self, masks_queries_logits: torch.Tensor, class_queries_logits: torch.Tensor, mask_labels: List[torch.Tensor], class_labels: List[torch.Tensor], auxiliary_predictions: Optional[Dict[str, torch.Tensor]]=None) -> Dict[str, torch.Tensor]:
        """
        This performs the loss computation.

        Args:
            masks_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, height, width)`.
            class_queries_logits (`torch.Tensor`):
                A tensor of shape `(batch_size, num_queries, num_labels)`.
            mask_labels (`torch.Tensor`):
                List of mask labels of shape `(labels, height, width)`.
            class_labels (`List[torch.Tensor]`):
                List of class labels of shape `(labels)`.
            auxiliary_predictions (`Dict[str, torch.Tensor]`, *optional*):
                if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], then it contains the logits from
                the inner layers of the Mask2FormerMaskedAttentionDecoder.

        Returns:
            losses (`Dict[str, Tensor]`): A dict of `torch.Tensor` containing three keys:
            - **loss_cross_entropy** -- The loss computed using cross entropy on the predicted and ground truth labels.
            - **loss_mask** -- The loss computed using sigmoid cross_entropy loss on the predicted and ground truth
              masks.
            - **loss_dice** -- The loss computed using dice loss on the predicted on the predicted and ground truth
              masks.
            if `use_auxiliary_loss` was set to `true` in [`Mask2FormerConfig`], the dictionary contains additional
            losses for each auxiliary predictions.
        """
        indices = self.matcher(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: Dict[str, Tensor] = {**self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks), **self.loss_labels(class_queries_logits, class_labels, indices)}
        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                masks_queries_logits = aux_outputs['masks_queries_logits']
                class_queries_logits = aux_outputs['class_queries_logits']
                loss_dict = self.forward(masks_queries_logits, class_queries_logits, mask_labels, class_labels)
                loss_dict = {f'{key}_{idx}': value for key, value in loss_dict.items()}
                losses.update(loss_dict)
        return losses

    def get_num_masks(self, class_labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Computes the average number of target masks across the batch, for normalization purposes.
        """
        num_masks = sum([len(classes) for classes in class_labels])
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)
        world_size = 1
        if is_accelerate_available():
            if PartialState._shared_state != {}:
                num_masks = reduce(num_masks)
                world_size = PartialState().num_processes
        num_masks = torch.clamp(num_masks / world_size, min=1)
        return num_masks