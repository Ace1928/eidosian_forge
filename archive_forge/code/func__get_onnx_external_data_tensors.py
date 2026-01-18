from pathlib import Path
from typing import List, Tuple, Union
import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors
def _get_onnx_external_data_tensors(model: onnx.ModelProto) -> List[str]:
    """
    Gets the paths of the external data tensors in the model.
    Note: make sure you load the model with load_external_data=False.
    """
    model_tensors = _get_initializer_tensors(model)
    model_tensors_ext = [ExternalDataInfo(tensor).location for tensor in model_tensors if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL]
    return model_tensors_ext