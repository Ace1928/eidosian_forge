import warnings
def _convert_conv_tranpose_padding_args_from_keras_to_torch(kernel_size, stride, dilation_rate, padding, output_padding):
    """Convert the padding arguments from Keras to the ones used by Torch.
    Torch starts with an output shape of `(input-1) * stride + kernel_size`,
    then removes `torch_padding` from both sides, and adds
    `torch_output_padding` on the right.
    Because in Torch the output_padding can only be added to the right,
    consistency with Tensorflow is not always possible. In particular this is
    the case when both the Torch padding and output_padding values are stricly
    positive.
    """
    assert padding.lower() in {'valid', 'same'}
    original_kernel_size = kernel_size
    kernel_size = (kernel_size - 1) * dilation_rate + 1
    if padding.lower() == 'valid':
        output_padding = max(kernel_size, stride) - kernel_size if output_padding is None else output_padding
        torch_padding = 0
        torch_output_padding = output_padding
    else:
        output_padding = stride - kernel_size % 2 if output_padding is None else output_padding
        torch_padding = max(-((kernel_size % 2 - kernel_size + output_padding) // 2), 0)
        torch_output_padding = 2 * torch_padding + kernel_size % 2 - kernel_size + output_padding
    if torch_padding > 0 and torch_output_padding > 0:
        warnings.warn(f'You might experience inconsistencies accross backends when calling conv transpose with kernel_size={original_kernel_size}, stride={stride}, dilation_rate={dilation_rate}, padding={padding}, output_padding={output_padding}.')
    if torch_output_padding >= stride:
        raise ValueError(f'The padding arguments (padding={padding}) and output_padding={output_padding}) lead to a Torch output_padding ({torch_output_padding}) that is greater than strides ({stride}). This is not supported. You can change the padding arguments, kernel or stride, or run on another backend. ')
    return (torch_padding, torch_output_padding)