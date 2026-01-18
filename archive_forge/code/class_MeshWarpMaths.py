import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo
class MeshWarpMaths(TVTKBaseInterface):
    """
    Performs the most basic mathematical operations on the warping field
    defined at each vertex of the input surface. A surface with scalar
    or vector data can be used as operator for non-uniform operations.

    .. warning:

      A point-to-point correspondence between surfaces is required


    Example::

        import nipype.algorithms.mesh as m
        mmath = m.MeshWarpMaths()
        mmath.inputs.in_surf = 'surf1.vtk'
        mmath.inputs.operator = 'surf2.vtk'
        mmath.inputs.operation = 'mul'
        res = mmath.run()

    """
    input_spec = MeshWarpMathsInputSpec
    output_spec = MeshWarpMathsOutputSpec

    def _run_interface(self, runtime):
        r1 = tvtk.PolyDataReader(file_name=self.inputs.in_surf)
        vtk1 = VTKInfo.vtk_output(r1)
        r1.update()
        points1 = np.array(vtk1.points)
        if vtk1.point_data.vectors is None:
            raise RuntimeError('No warping field was found in in_surf')
        operator = self.inputs.operator
        opfield = np.ones_like(points1)
        if isinstance(operator, (str, bytes)):
            r2 = tvtk.PolyDataReader(file_name=self.inputs.surface2)
            vtk2 = VTKInfo.vtk_output(r2)
            r2.update()
            assert len(points1) == len(vtk2.points)
            opfield = vtk2.point_data.vectors
            if opfield is None:
                opfield = vtk2.point_data.scalars
            if opfield is None:
                raise RuntimeError('No operator values found in operator file')
            opfield = np.array(opfield)
            if opfield.shape[1] < points1.shape[1]:
                opfield = np.array([opfield.tolist()] * points1.shape[1]).T
        else:
            operator = np.atleast_1d(operator)
            opfield *= operator
        warping = np.array(vtk1.point_data.vectors)
        if self.inputs.operation == 'sum':
            warping += opfield
        elif self.inputs.operation == 'sub':
            warping -= opfield
        elif self.inputs.operation == 'mul':
            warping *= opfield
        elif self.inputs.operation == 'div':
            warping /= opfield
        vtk1.point_data.vectors = warping
        writer = tvtk.PolyDataWriter(file_name=op.abspath(self.inputs.out_warp))
        VTKInfo.configure_input_data(writer, vtk1)
        writer.write()
        vtk1.point_data.vectors = None
        vtk1.points = points1 + warping
        writer = tvtk.PolyDataWriter(file_name=op.abspath(self.inputs.out_file))
        VTKInfo.configure_input_data(writer, vtk1)
        writer.write()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        outputs['out_warp'] = op.abspath(self.inputs.out_warp)
        return outputs