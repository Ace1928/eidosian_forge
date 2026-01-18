import os
import os.path as op
import re
from glob import glob
import tempfile
import numpy as np
from ...utils.filemanip import load_json, save_json, split_filename, fname_presuffix
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
class WarpPoints(CommandLine):
    """Use FSL `img2imgcoord <http://fsl.fmrib.ox.ac.uk/fsl/fsl-4.1.9/flirt/overview.html>`_
    to transform point sets. Accepts plain text files and vtk files.

    .. Note:: transformation of TrackVis trk files is not yet implemented


    Examples
    --------

    >>> from nipype.interfaces.fsl import WarpPoints
    >>> warppoints = WarpPoints()
    >>> warppoints.inputs.in_coords = 'surf.txt'
    >>> warppoints.inputs.src_file = 'epi.nii'
    >>> warppoints.inputs.dest_file = 'T1.nii'
    >>> warppoints.inputs.warp_file = 'warpfield.nii'
    >>> warppoints.inputs.coord_mm = True
    >>> warppoints.cmdline # doctest: +ELLIPSIS
    'img2imgcoord -mm -dest T1.nii -src epi.nii -warp warpfield.nii surf.txt'
    >>> res = warppoints.run() # doctest: +SKIP


    """
    input_spec = WarpPointsInputSpec
    output_spec = WarpPointsOutputSpec
    _cmd = 'img2imgcoord'
    _terminal_output = 'stream'

    def __init__(self, command=None, **inputs):
        self._tmpfile = None
        self._in_file = None
        self._outformat = None
        super(WarpPoints, self).__init__(command=command, **inputs)

    def _format_arg(self, name, trait_spec, value):
        if name == 'out_file':
            return ''
        return super(WarpPoints, self)._format_arg(name, trait_spec, value)

    def _parse_inputs(self, skip=None):
        fname, ext = op.splitext(self.inputs.in_coords)
        setattr(self, '_in_file', fname)
        setattr(self, '_outformat', ext[1:])
        first_args = super(WarpPoints, self)._parse_inputs(skip=['in_coords', 'out_file'])
        second_args = fname + '.txt'
        if ext in ['.vtk', '.trk']:
            if self._tmpfile is None:
                self._tmpfile = tempfile.NamedTemporaryFile(suffix='.txt', dir=os.getcwd(), delete=False).name
            second_args = self._tmpfile
        return first_args + [second_args]

    def _vtk_to_coords(self, in_file, out_file=None):
        from ..vtkbase import tvtk
        from ...interfaces import vtkbase as VTKInfo
        if VTKInfo.no_tvtk():
            raise ImportError('TVTK is required and tvtk package was not found')
        reader = tvtk.PolyDataReader(file_name=in_file + '.vtk')
        reader.update()
        mesh = VTKInfo.vtk_output(reader)
        points = mesh.points
        if out_file is None:
            out_file, _ = op.splitext(in_file) + '.txt'
        np.savetxt(out_file, points)
        return out_file

    def _coords_to_vtk(self, points, out_file):
        from ..vtkbase import tvtk
        from ...interfaces import vtkbase as VTKInfo
        if VTKInfo.no_tvtk():
            raise ImportError('TVTK is required and tvtk package was not found')
        reader = tvtk.PolyDataReader(file_name=self.inputs.in_file)
        reader.update()
        mesh = VTKInfo.vtk_output(reader)
        mesh.points = points
        writer = tvtk.PolyDataWriter(file_name=out_file)
        VTKInfo.configure_input_data(writer, mesh)
        writer.write()

    def _trk_to_coords(self, in_file, out_file=None):
        from nibabel.trackvis import TrackvisFile
        trkfile = TrackvisFile.from_file(in_file)
        streamlines = trkfile.streamlines
        if out_file is None:
            out_file, _ = op.splitext(in_file)
        np.savetxt(out_file + '.txt', streamlines)
        return out_file + '.txt'

    def _coords_to_trk(self, points, out_file):
        raise NotImplementedError('trk files are not yet supported')

    def _overload_extension(self, value, name):
        if name == 'out_file':
            return '%s.%s' % (value, getattr(self, '_outformat'))

    def _run_interface(self, runtime):
        fname = getattr(self, '_in_file')
        outformat = getattr(self, '_outformat')
        tmpfile = None
        if outformat == 'vtk':
            tmpfile = self._tmpfile
            self._vtk_to_coords(fname, out_file=tmpfile)
        elif outformat == 'trk':
            tmpfile = self._tmpfile
            self._trk_to_coords(fname, out_file=tmpfile)
        runtime = super(WarpPoints, self)._run_interface(runtime)
        newpoints = np.fromstring('\n'.join(runtime.stdout.split('\n')[1:]), sep=' ')
        if tmpfile is not None:
            try:
                os.remove(tmpfile.name)
            except:
                pass
        out_file = self._filename_from_source('out_file')
        if outformat == 'vtk':
            self._coords_to_vtk(newpoints, out_file)
        elif outformat == 'trk':
            self._coords_to_trk(newpoints, out_file)
        else:
            np.savetxt(out_file, newpoints.reshape(-1, 3))
        return runtime