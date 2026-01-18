from ...configuration_utils import PretrainedConfig
from ...utils import logging

    This is the configuration class to store the configuration for a timm backbone [`TimmBackbone`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone (`str`, *optional*):
            The timm checkpoint to load.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        features_only (`bool`, *optional*, defaults to `True`):
            Whether to output only the features or also the logits.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use a pretrained backbone.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). Will default to the last stage if unset.
        freeze_batch_norm_2d (`bool`, *optional*, defaults to `False`):
            Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.

    Example:
    ```python
    >>> from transformers import TimmBackboneConfig, TimmBackbone

    >>> # Initializing a timm backbone
    >>> configuration = TimmBackboneConfig("resnet50")

    >>> # Initializing a model from the configuration
    >>> model = TimmBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    