import inspect
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ..utils import (
def _dispatch_accelerate_model(self, device_map: str, max_memory: Optional[int]=None, offload_folder: Optional[str]=None, offload_index: Optional[int]=None) -> None:
    """
        Optional re-dispatch the model and attach new hooks to the model in case the model has been loaded with
        accelerate (i.e. with `device_map=xxx`)

        Args:
            device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_index (`int`, *optional*):
                The offload_index argument to be passed to `accelerate.dispatch_model` method.
        """
    dispatch_model_kwargs = {}
    if 'offload_index' in inspect.signature(dispatch_model).parameters:
        dispatch_model_kwargs['offload_index'] = offload_index
    no_split_module_classes = self._no_split_modules
    if device_map != 'sequential':
        max_memory = get_balanced_memory(self, max_memory=max_memory, no_split_module_classes=no_split_module_classes, low_zero=device_map == 'balanced_low_0')
    if isinstance(device_map, str):
        device_map = infer_auto_device_map(self, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    dispatch_model(self, device_map=device_map, offload_dir=offload_folder, **dispatch_model_kwargs)