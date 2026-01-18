from math import ceil
import cupy
class _UpFIRDn(object):

    def __init__(self, h, x_dtype, up, down):
        """Helper for resampling"""
        h = cupy.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError('h must be 1D with non-zero length')
        self._output_type = cupy.result_type(h.dtype, x_dtype, cupy.float32)
        h = cupy.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError('Both up and down must be >= 1')
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cupy.asarray(self._h_trans_flip)
        self._h_trans_flip = cupy.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(self, x, axis):
        """Apply the prepared filter to the specified axis of a nD signal x"""
        x = cupy.asarray(x, self._output_type)
        output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        out = cupy.empty(output_shape, dtype=self._output_type, order='C')
        axis = axis % x.ndim
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + len(self._h_trans_flip) // self._up - 1
        if out.ndim == 1:
            threadsperblock, blockspergrid = _get_tpb_bpg()
            kernel = UPFIRDN_MODULE.get_function(f'_cupy_upfirdn1D_{out.dtype.name}')
            kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (x, self._h_trans_flip, self._up, self._down, axis, x_shape_a, h_per_phase, padded_len, out, out.shape[0]))
        elif out.ndim == 2:
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()
            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            kernel = UPFIRDN_MODULE.get_function(f'_cupy_upfirdn2D_{out.dtype.name}')
            kernel(threadsperblock, blockspergrid, (x, x.shape[1], self._h_trans_flip, self._up, self._down, axis, x_shape_a, h_per_phase, padded_len, out, out.shape[0], out.shape[1]))
        else:
            raise NotImplementedError('upfirdn() requires ndim <= 2')
        return out