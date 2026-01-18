import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Tuple
import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F
from ._transform import _RandomApplyTransform, Transform
from ._utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size
class MixUp(_BaseMixUpCutMix):
    """[BETA] Apply MixUp to the provided batch of images and labels.

    .. v2betastatus:: MixUp transform

    Paper: `mixup: Beyond Empirical Risk Minimization <https://arxiv.org/abs/1710.09412>`_.

    .. note::
        This transform is meant to be used on **batches** of samples, not
        individual images. See
        :ref:`sphx_glr_auto_examples_transforms_plot_cutmix_mixup.py` for detailed usage
        examples.
        The sample pairing is deterministic and done by matching consecutive
        samples in the batch, so the batch needs to be shuffled (this is an
        implementation detail, not a guaranteed convention.)

    In the input, the labels are expected to be a tensor of shape ``(batch_size,)``. They will be transformed
    into a tensor of shape ``(batch_size, num_classes)``.

    Args:
        alpha (float, optional): hyperparameter of the Beta distribution used for mixup. Default is 1.
        num_classes (int): number of classes in the batch. Used for one-hot-encoding.
        labels_getter (callable or "default", optional): indicates how to identify the labels in the input.
            By default, this will pick the second parameter as the labels if it's a tensor. This covers the most
            common scenario where this transform is called as ``MixUp()(imgs_batch, labels_batch)``.
            It can also be a callable that takes the same input as the transform, and returns the labels.
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict(lam=float(self._dist.sample(())))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        lam = params['lam']
        if inpt is params['labels']:
            return self._mixup_label(inpt, lam=lam)
        elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)) or is_pure_tensor(inpt):
            self._check_image_or_video(inpt, batch_size=params['batch_size'])
            output = inpt.roll(1, 0).mul_(1.0 - lam).add_(inpt.mul(lam))
            if isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
                output = tv_tensors.wrap(output, like=inpt)
            return output
        else:
            return inpt