from keras.src import backend
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import any_symbolic_tensors
from keras.src.ops.operation import Operation
from keras.src.ops.operation_utils import compute_conv_output_shape
def _crop_images(images, top_cropping, bottom_cropping, left_cropping, right_cropping, target_height, target_width):
    images = backend.convert_to_tensor(images)
    is_batch = True
    images_shape = ops.shape(images)
    if len(images_shape) == 3:
        is_batch = False
        images = backend.numpy.expand_dims(images, 0)
    elif len(images_shape) != 4:
        raise ValueError(f'Invalid shape for argument `images`: it must have rank 3 or 4. Received: images.shape={images_shape}')
    batch, height, width, depth = ops.shape(images)
    if [top_cropping, bottom_cropping, target_height].count(None) != 1:
        raise ValueError(f'Must specify exactly two of top_cropping, bottom_cropping, target_height. Received: top_cropping={top_cropping}, bottom_cropping={bottom_cropping}, target_height={target_height}')
    if [left_cropping, right_cropping, target_width].count(None) != 1:
        raise ValueError(f'Must specify exactly two of left_cropping, right_cropping, target_width. Received: left_cropping={left_cropping}, right_cropping={right_cropping}, target_width={target_width}')
    if top_cropping is None:
        top_cropping = height - target_height - bottom_cropping
    if target_height is None:
        target_height = height - bottom_cropping - top_cropping
    if left_cropping is None:
        left_cropping = width - target_width - right_cropping
    if target_width is None:
        target_width = width - right_cropping - left_cropping
    if top_cropping < 0:
        raise ValueError(f'top_cropping must be >= 0. Received: top_cropping={top_cropping}')
    if target_height < 0:
        raise ValueError(f'target_height must be >= 0. Received: target_height={target_height}')
    if left_cropping < 0:
        raise ValueError(f'left_cropping must be >= 0. Received: left_cropping={left_cropping}')
    if target_width < 0:
        raise ValueError(f'target_width must be >= 0. Received: target_width={target_width}')
    cropped = ops.slice(images, backend.numpy.stack([0, top_cropping, left_cropping, 0]), backend.numpy.stack([batch, target_height, target_width, depth]))
    cropped_shape = [batch, target_height, target_width, depth]
    cropped = backend.numpy.reshape(cropped, cropped_shape)
    if not is_batch:
        cropped = backend.numpy.squeeze(cropped, axis=[0])
    return cropped