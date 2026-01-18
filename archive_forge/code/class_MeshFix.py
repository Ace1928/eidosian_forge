import os.path as op
from ..utils.filemanip import split_filename
from .base import (
class MeshFix(CommandLine):
    """
    MeshFix v1.2-alpha - by Marco Attene, Mirko Windhoff, Axel Thielscher.

    .. seealso::

        http://jmeshlib.sourceforge.net
            Sourceforge page

        http://simnibs.de/installation/meshfixandgetfem
            Ubuntu installation instructions

    If MeshFix is used for research purposes, please cite the following paper:
    M. Attene - A lightweight approach to repairing digitized polygon meshes.
    The Visual Computer, 2010. (c) Springer.

    Accepted input formats are OFF, PLY and STL.
    Other formats (like .msh for gmsh) are supported only partially.

    Example
    -------

    >>> import nipype.interfaces.meshfix as mf
    >>> fix = mf.MeshFix()
    >>> fix.inputs.in_file1 = 'lh-pial.stl'
    >>> fix.inputs.in_file2 = 'rh-pial.stl'
    >>> fix.run()                                    # doctest: +SKIP
    >>> fix.cmdline
    'meshfix lh-pial.stl rh-pial.stl -o lh-pial_fixed.off'
    """
    _cmd = 'meshfix'
    input_spec = MeshFixInputSpec
    output_spec = MeshFixOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if isdefined(self.inputs.out_filename):
            path, name, ext = split_filename(self.inputs.out_filename)
            ext = ext.replace('.', '')
            out_types = ['stl', 'msh', 'wrl', 'vrml', 'fs', 'off']
            if any((ext == out_type.lower() for out_type in out_types)):
                outputs['mesh_file'] = op.abspath(self.inputs.out_filename)
            else:
                outputs['mesh_file'] = op.abspath(name + '.' + self.inputs.output_type)
        else:
            outputs['mesh_file'] = op.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file1)
        if self.inputs.save_as_freesurfer_mesh or self.inputs.output_type == 'fs':
            self.inputs.output_type = 'fs'
            self.inputs.save_as_freesurfer_mesh = True
        if self.inputs.save_as_stl or self.inputs.output_type == 'stl':
            self.inputs.output_type = 'stl'
            self.inputs.save_as_stl = True
        if self.inputs.save_as_vrml or self.inputs.output_type == 'vrml':
            self.inputs.output_type = 'vrml'
            self.inputs.save_as_vrml = True
        return name + '_fixed.' + self.inputs.output_type