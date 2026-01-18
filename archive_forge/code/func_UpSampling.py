from ._internal import NDArrayBase
from ..base import _Null
def UpSampling(*data, **kwargs):
    """Upsamples the given input data.

    Two algorithms (``sample_type``) are available for upsampling:

    - Nearest Neighbor
    - Bilinear

    **Nearest Neighbor Upsampling**

    Input data is expected to be NCHW.

    Example::

      x = [[[[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]]]

      UpSampling(x, scale=2, sample_type='nearest') = [[[[1. 1. 1. 1. 1. 1.]
                                                         [1. 1. 1. 1. 1. 1.]
                                                         [1. 1. 1. 1. 1. 1.]
                                                         [1. 1. 1. 1. 1. 1.]
                                                         [1. 1. 1. 1. 1. 1.]
                                                         [1. 1. 1. 1. 1. 1.]]]]

    **Bilinear Upsampling**

    Uses `deconvolution` algorithm under the hood. You need provide both input data and the kernel.

    Input data is expected to be NCHW.

    `num_filter` is expected to be same as the number of channels.

    Example::

      x = [[[[1. 1. 1.]
             [1. 1. 1.]
             [1. 1. 1.]]]]

      w = [[[[1. 1. 1. 1.]
             [1. 1. 1. 1.]
             [1. 1. 1. 1.]
             [1. 1. 1. 1.]]]]
  
      UpSampling(x, w, scale=2, sample_type='bilinear', num_filter=1) = [[[[1. 2. 2. 2. 2. 1.]
                                                                           [2. 4. 4. 4. 4. 2.]
                                                                           [2. 4. 4. 4. 4. 2.]
                                                                           [2. 4. 4. 4. 4. 2.]
                                                                           [2. 4. 4. 4. 4. 2.]
                                                                           [1. 2. 2. 2. 2. 1.]]]]


    Defined in ../src/operator/nn/upsampling.cc:L172

    Parameters
    ----------
    data : NDArray[]
        Array of tensors to upsample. For bilinear upsampling, there should be 2 inputs - 1 data and 1 weight.
    scale : int, required
        Up sampling scale
    num_filter : int, optional, default='0'
        Input filter. Only used by bilinear sample_type.Since bilinear upsampling uses deconvolution, num_filters is set to the number of channels.
    sample_type : {'bilinear', 'nearest'}, required
        upsampling method
    multi_input_mode : {'concat', 'sum'},optional, default='concat'
        How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
    workspace : long (non-negative), optional, default=512
        Tmp workspace for deconvolution (MB)

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return (0,)