from __future__ import annotations
import io
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import _image_decoder_data, expect
def _generate_test_data(format_: str, frozen_data: _image_decoder_data.ImageDecoderData, pixel_format: str='RGB', height: int=32, width: int=32, tile_sz: int=5) -> tuple[np.ndarray, np.ndarray]:
    try:
        import PIL.Image
    except ImportError:
        return (frozen_data.data, frozen_data.output)
    np.random.seed(12345)
    image = generate_checkerboard(height, width, tile_sz)
    image_pil = PIL.Image.fromarray(image)
    with io.BytesIO() as f:
        image_pil.save(f, format=format_)
        data = f.getvalue()
        data_array = np.frombuffer(data, dtype=np.uint8)
    if pixel_format == 'BGR':
        output_pil = PIL.Image.open(io.BytesIO(data))
        output = np.array(output_pil)[:, :, ::-1]
    elif pixel_format == 'RGB':
        output_pil = PIL.Image.open(io.BytesIO(data))
        output = np.array(output_pil)
    elif pixel_format == 'Grayscale':
        output_pil = PIL.Image.open(io.BytesIO(data)).convert('L')
        output = np.array(output_pil)[:, :, np.newaxis]
    else:
        raise ValueError(f'Unsupported pixel format: {pixel_format}')
    return (data_array, output)