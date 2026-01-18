import warnings
def _convert_conv_tranpose_padding_args_from_keras_to_jax(kernel_size, stride, dilation_rate, padding, output_padding):
    """Convert the padding arguments from Keras to the ones used by JAX.
    JAX starts with an shape of size `(input-1) * stride - kernel_size + 2`,
    then adds `left_pad` on the left, and `right_pad` on the right.
    In Keras, the `padding` argument determines a base shape, to which
    `output_padding` is added on the right. If `output_padding` is None, it will
    be given a default value.
    """
    assert padding.lower() in {'valid', 'same'}
    kernel_size = (kernel_size - 1) * dilation_rate + 1
    if padding.lower() == 'valid':
        output_padding = max(kernel_size, stride) - kernel_size if output_padding is None else output_padding
        left_pad = kernel_size - 1
        right_pad = kernel_size - 1 + output_padding
    else:
        if output_padding is None:
            pad_len = stride + kernel_size - 2
        else:
            pad_len = kernel_size + kernel_size % 2 - 2 + output_padding
        left_pad = min(pad_len // 2 + pad_len % 2, kernel_size - 1)
        right_pad = pad_len - left_pad
    return (left_pad, right_pad)