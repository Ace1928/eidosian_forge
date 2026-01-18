import cupy
from cupy import _core
from cupy.cuda import texture
from cupy.cuda import runtime
def _create_texture_object(data, address_mode: str, filter_mode: str, read_mode: str, border_color=0):
    if cupy.issubdtype(data.dtype, cupy.unsignedinteger):
        fmt_kind = runtime.cudaChannelFormatKindUnsigned
    elif cupy.issubdtype(data.dtype, cupy.integer):
        fmt_kind = runtime.cudaChannelFormatKindSigned
    elif cupy.issubdtype(data.dtype, cupy.floating):
        fmt_kind = runtime.cudaChannelFormatKindFloat
    else:
        raise ValueError(f'Unsupported data type {data.dtype}')
    if address_mode == 'nearest':
        address_mode = runtime.cudaAddressModeClamp
    elif address_mode == 'constant':
        address_mode = runtime.cudaAddressModeBorder
    else:
        raise ValueError(f'Unsupported address mode {address_mode} (supported: constant, nearest)')
    if filter_mode == 'nearest':
        filter_mode = runtime.cudaFilterModePoint
    elif filter_mode == 'linear':
        filter_mode = runtime.cudaFilterModeLinear
    else:
        raise ValueError(f'Unsupported filter mode {filter_mode} (supported: nearest, linear)')
    if read_mode == 'element_type':
        read_mode = runtime.cudaReadModeElementType
    elif read_mode == 'normalized_float':
        read_mode = runtime.cudaReadModeNormalizedFloat
    else:
        raise ValueError(f'Unsupported read mode {read_mode} (supported: element_type, normalized_float)')
    texture_fmt = texture.ChannelFormatDescriptor(data.itemsize * 8, 0, 0, 0, fmt_kind)
    array = texture.CUDAarray(texture_fmt, *data.shape[::-1])
    res_desc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=array)
    tex_desc = texture.TextureDescriptor((address_mode,) * data.ndim, filter_mode, read_mode, borderColors=(border_color,))
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    array.copy_from(data)
    return tex_obj