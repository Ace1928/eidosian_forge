import os
import os.path as op
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import (
from ... import logging
class TShift(AFNICommand):
    """Shifts voxel time series from input so that separate slices are aligned
    to the same temporal origin.

    For complete details, see the `3dTshift Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dTshift.html>`_

    Examples
    --------
    Slice timing details may be specified explicitly via the ``slice_timing``
    input:

    >>> from nipype.interfaces import afni
    >>> TR = 2.5
    >>> tshift = afni.TShift()
    >>> tshift.inputs.in_file = 'functional.nii'
    >>> tshift.inputs.tzero = 0.0
    >>> tshift.inputs.tr = '%.1fs' % TR
    >>> tshift.inputs.slice_timing = list(np.arange(40) / TR)
    >>> tshift.cmdline
    '3dTshift -prefix functional_tshift -tpattern @slice_timing.1D -TR 2.5s -tzero 0.0 functional.nii'

    When the ``slice_timing`` input is used, the ``timing_file`` output is populated,
    in this case with the generated file.

    >>> tshift._list_outputs()['timing_file']  # doctest: +ELLIPSIS
    '.../slice_timing.1D'

    >>> np.loadtxt(tshift._list_outputs()['timing_file']).tolist()[:5]
    [0.0, 0.4, 0.8, 1.2, 1.6]

    If ``slice_encoding_direction`` is set to ``'k-'``, the slice timing is reversed:

    >>> tshift.inputs.slice_encoding_direction = 'k-'
    >>> tshift.cmdline
    '3dTshift -prefix functional_tshift -tpattern @slice_timing.1D -TR 2.5s -tzero 0.0 functional.nii'
    >>> np.loadtxt(tshift._list_outputs()['timing_file']).tolist()[:5]
    [15.6, 15.2, 14.8, 14.4, 14.0]

    This method creates a ``slice_timing.1D`` file to be passed to ``3dTshift``.
    A pre-existing slice-timing file may be used in the same way:

    >>> tshift = afni.TShift()
    >>> tshift.inputs.in_file = 'functional.nii'
    >>> tshift.inputs.tzero = 0.0
    >>> tshift.inputs.tr = '%.1fs' % TR
    >>> tshift.inputs.slice_timing = 'slice_timing.1D'
    >>> tshift.cmdline
    '3dTshift -prefix functional_tshift -tpattern @slice_timing.1D -TR 2.5s -tzero 0.0 functional.nii'

    When a pre-existing file is provided, ``timing_file`` is simply passed through.

    >>> tshift._list_outputs()['timing_file']  # doctest: +ELLIPSIS
    '.../slice_timing.1D'

    Alternatively, pre-specified slice timing patterns may be specified with the
    ``tpattern`` input.
    For example, to specify an alternating, ascending slice timing pattern:

    >>> tshift = afni.TShift()
    >>> tshift.inputs.in_file = 'functional.nii'
    >>> tshift.inputs.tzero = 0.0
    >>> tshift.inputs.tr = '%.1fs' % TR
    >>> tshift.inputs.tpattern = 'alt+z'
    >>> tshift.cmdline
    '3dTshift -prefix functional_tshift -tpattern alt+z -TR 2.5s -tzero 0.0 functional.nii'

    For backwards compatibility, ``tpattern`` may also take filenames prefixed
    with ``@``.
    However, in this case, filenames are not validated, so this usage will be
    deprecated in future versions of Nipype.

    >>> tshift = afni.TShift()
    >>> tshift.inputs.in_file = 'functional.nii'
    >>> tshift.inputs.tzero = 0.0
    >>> tshift.inputs.tr = '%.1fs' % TR
    >>> tshift.inputs.tpattern = '@slice_timing.1D'
    >>> tshift.cmdline
    '3dTshift -prefix functional_tshift -tpattern @slice_timing.1D -TR 2.5s -tzero 0.0 functional.nii'

    In these cases, ``timing_file`` is undefined.

    >>> tshift._list_outputs()['timing_file']  # doctest: +ELLIPSIS
    <undefined>

    In any configuration, the interface may be run as usual:

    >>> res = tshift.run()  # doctest: +SKIP
    """
    _cmd = '3dTshift'
    input_spec = TShiftInputSpec
    output_spec = TShiftOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if name == 'tpattern' and value.startswith('@'):
            iflogger.warning('Passing a file prefixed by "@" will be deprecated; please use the `slice_timing` input')
        elif name == 'slice_timing' and isinstance(value, list):
            value = self._write_slice_timing()
        return super(TShift, self)._format_arg(name, trait_spec, value)

    def _write_slice_timing(self):
        slice_timing = list(self.inputs.slice_timing)
        if self.inputs.slice_encoding_direction.endswith('-'):
            slice_timing.reverse()
        fname = 'slice_timing.1D'
        with open(fname, 'w') as fobj:
            fobj.write('\t'.join(map(str, slice_timing)))
        return fname

    def _list_outputs(self):
        outputs = super(TShift, self)._list_outputs()
        if isdefined(self.inputs.slice_timing):
            if isinstance(self.inputs.slice_timing, list):
                outputs['timing_file'] = os.path.abspath('slice_timing.1D')
            else:
                outputs['timing_file'] = os.path.abspath(self.inputs.slice_timing)
        return outputs