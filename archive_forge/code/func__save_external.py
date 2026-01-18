from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def _save_external(self, file_path: str, all_tensors_to_one_file: bool) -> onnx.ModelProto:
    """Save the large model into a main onnx file and one file
        per tensor. Follows the same format as :func:`write_external_data_tensors
        <onnx.external_data_helper.write_external_data_tensors>`.
        The main model needs to be modified to update the file location,
        the function returns this modified copy.

        Arguments:
            file_path: model file
            all_tensors_to_one_file: all tensors in one file

        Returns:
            modified main model proto
        """

    def _clean_name(prefix: str, name: str, unique_names: dict[str, int]) -> str:
        if prefix:
            name = f'{prefix}-{name}'
        for c in ':/\\;,!':
            name = name.replace(c, '')
        base_name = name
        if name in unique_names:
            i = unique_names[name] + 1
            unique_names[name] = i
            return f'{base_name}_{i}'
        unique_names[name] = 1
        return name
    unique_names: dict[str, int] = {}
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Folder {folder!r} does not exist.')
    proto = self.model_proto.SerializeToString()
    copy = onnx.ModelProto()
    copy.ParseFromString(proto)
    prefix = os.path.splitext(os.path.split(file_path)[-1])[0]
    if all_tensors_to_one_file:
        file_weight = f'{os.path.split(file_path)[1]}.weight'
        full_file_weight = f'{file_path}.weight'
        offset = 0
        with open(full_file_weight, 'wb') as f:
            pass
    for tensor in ext_data._get_all_tensors(copy):
        if not ext_data.uses_external_data(tensor):
            continue
        prop: onnx.StringStringEntryProto | None = None
        for ext in tensor.external_data:
            if ext.key == 'location':
                prop = ext
        if prop is None:
            raise RuntimeError(f'No location found for tensor name {tensor.name!r}.')
        if prop.value not in self.large_initializers:
            raise RuntimeError(f'Unable to find large tensor named {tensor.name!r} with location {prop.value!r} in {sorted(self.large_initializers)}.')
        np_tensor = self.large_initializers[prop.value]
        if sys.byteorder == 'big':
            tensor_bytes = np_tensor.byteswap().tobytes()
        else:
            tensor_bytes = np_tensor.tobytes()
        if all_tensors_to_one_file:
            _set_external_data(tensor, location=file_weight, offset=offset, length=len(tensor_bytes))
            offset += len(tensor_bytes)
            with open(full_file_weight, 'ab') as f:
                f.write(tensor_bytes)
        else:
            name = f'{_clean_name(prefix, prop.value, unique_names)}.weight'
            _set_external_data(tensor, location=name)
            full_name = os.path.join(folder, name)
            prop.value = name
            with open(full_name, 'wb') as f:
                f.write(tensor_bytes)
    with open(file_path, 'wb') as f:
        f.write(copy.SerializeToString())
    return copy