import os
import os.path as op
import re
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename
from ..base import (
from ...external.due import BibTeX
from .base import (
class NwarpCat(AFNICommand):
    """Catenates (composes) 3D warps defined on a grid, OR via a matrix.

    .. note::

      * All transformations are from DICOM xyz (in mm) to DICOM xyz.

      * Matrix warps are in files that end in '.1D' or in '.txt'.  A matrix
        warp file should have 12 numbers in it, as output (for example), by
        '3dAllineate -1Dmatrix_save'.

      * Nonlinear warps are in dataset files (AFNI .HEAD/.BRIK or NIfTI .nii)
        with 3 sub-bricks giving the DICOM order xyz grid displacements in mm.

      * If all the input warps are matrices, then the output is a matrix
        and will be written to the file 'prefix.aff12.1D'.
        Unless the prefix already contains the string '.1D', in which case
        the filename is just the prefix.

      * If 'prefix' is just 'stdout', then the output matrix is written
        to standard output.
        In any of these cases, the output format is 12 numbers in one row.

      * If any of the input warps are datasets, they must all be defined on
        the same 3D grid!
        And of course, then the output will be a dataset on the same grid.
        However, you can expand the grid using the '-expad' option.

      * The order of operations in the final (output) warp is, for the
        case of 3 input warps:

            OUTPUT(x) = warp3( warp2( warp1(x) ) )

       That is, warp1 is applied first, then warp2, et cetera.
       The 3D x coordinates are taken from each grid location in the
       first dataset defined on a grid.

    For complete details, see the `3dNwarpCat Documentation.
    <https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNwarpCat.html>`_

    Examples
    --------
    >>> from nipype.interfaces import afni
    >>> nwarpcat = afni.NwarpCat()
    >>> nwarpcat.inputs.in_files = ['Q25_warp+tlrc.HEAD', ('IDENT', 'structural.nii')]
    >>> nwarpcat.inputs.out_file = 'Fred_total_WARP'
    >>> nwarpcat.cmdline
    "3dNwarpCat -interp wsinc5 -prefix Fred_total_WARP Q25_warp+tlrc.HEAD 'IDENT(structural.nii)'"
    >>> res = nwarpcat.run()  # doctest: +SKIP

    """
    _cmd = '3dNwarpCat'
    input_spec = NwarpCatInputSpec
    output_spec = AFNICommandOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'in_files':
            return spec.argstr % ' '.join(["'" + v[0] + '(' + v[1] + ")'" if isinstance(v, tuple) else v for v in value])
        return super(NwarpCat, self)._format_arg(name, spec, value)

    def _gen_filename(self, name):
        if name == 'out_file':
            return self._gen_fname(self.inputs.in_files[0][0], suffix='_NwarpCat')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_file):
            outputs['out_file'] = os.path.abspath(self.inputs.out_file)
        else:
            outputs['out_file'] = os.path.abspath(self._gen_fname(self.inputs.in_files[0], suffix='_NwarpCat+tlrc', ext='.HEAD'))
        return outputs