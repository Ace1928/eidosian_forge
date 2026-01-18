from pathlib import Path
from typing import List, Tuple, Union
import onnx
from onnx.external_data_helper import ExternalDataInfo, _get_initializer_tensors
def _get_external_data_paths(src_paths: List[Path], dst_paths: List[Path]) -> Tuple[List[Path], List[str]]:
    """
    Gets external data paths from the model and add them to the list of files to copy.
    """
    model_paths = src_paths.copy()
    for idx, model_path in enumerate(model_paths):
        model = onnx.load(str(model_path), load_external_data=False)
        model_tensors = _get_initializer_tensors(model)
        model_tensors_ext = [ExternalDataInfo(tensor).location for tensor in model_tensors if tensor.HasField('data_location') and tensor.data_location == onnx.TensorProto.EXTERNAL]
        if len(set(model_tensors_ext)) == 1:
            src_paths.append(model_path.parent / model_tensors_ext[0])
            dst_paths.append(dst_paths[idx].parent / model_tensors_ext[0])
        else:
            src_paths.extend([model_path.parent / tensor_name for tensor_name in model_tensors_ext])
            dst_paths.extend((dst_paths[idx].parent / tensor_name for tensor_name in model_tensors_ext))
    return (src_paths, dst_paths)