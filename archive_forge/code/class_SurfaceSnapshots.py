import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
class SurfaceSnapshots(FSCommand):
    """Use Tksurfer to save pictures of the cortical surface.

    By default, this takes snapshots of the lateral, medial, ventral,
    and dorsal surfaces.  See the ``six_images`` option to add the
    anterior and posterior surfaces.

    You may also supply your own tcl script (see the Freesurfer wiki for
    information on scripting tksurfer). The screenshot stem is set as the
    environment variable "_SNAPSHOT_STEM", which you can use in your
    own scripts.

    Node that this interface will not run if you do not have graphics
    enabled on your system.

    Examples
    --------

    >>> import nipype.interfaces.freesurfer as fs
    >>> shots = fs.SurfaceSnapshots(subject_id="fsaverage", hemi="lh", surface="pial")
    >>> shots.inputs.overlay = "zstat1.nii.gz"
    >>> shots.inputs.overlay_range = (2.3, 6)
    >>> shots.inputs.overlay_reg = "register.dat"
    >>> res = shots.run() # doctest: +SKIP

    """
    _cmd = 'tksurfer'
    input_spec = SurfaceSnapshotsInputSpec
    output_spec = SurfaceSnapshotsOutputSpec

    def _format_arg(self, name, spec, value):
        if name == 'tcl_script':
            if not isdefined(value):
                return '-tcl snapshots.tcl'
            else:
                return '-tcl %s' % value
        elif name == 'overlay_range':
            if isinstance(value, float):
                return '-fthresh %.3f' % value
            elif len(value) == 2:
                return '-fminmax %.3f %.3f' % value
            else:
                return '-fminmax %.3f %.3f -fmid %.3f' % (value[0], value[2], value[1])
        elif name == 'annot_name' and isdefined(value):
            if value.endswith('.annot'):
                value = value[:-6]
            if re.match('%s[\\.\\-_]' % self.inputs.hemi, value[:3]):
                value = value[3:]
            return '-annotation %s' % value
        return super(SurfaceSnapshots, self)._format_arg(name, spec, value)

    def _run_interface(self, runtime):
        if not isdefined(self.inputs.screenshot_stem):
            stem = '%s_%s_%s' % (self.inputs.subject_id, self.inputs.hemi, self.inputs.surface)
        else:
            stem = self.inputs.screenshot_stem
            stem_args = self.inputs.stem_template_args
            if isdefined(stem_args):
                args = tuple([getattr(self.inputs, arg) for arg in stem_args])
                stem = stem % args
        if 'DISPLAY' not in os.environ:
            raise RuntimeError('Graphics are not enabled -- cannot run tksurfer')
        runtime.environ['_SNAPSHOT_STEM'] = stem
        self._write_tcl_script()
        runtime = super(SurfaceSnapshots, self)._run_interface(runtime)
        errors = ['surfer: failed, no suitable display found', 'Fatal Error in tksurfer.bin: could not open display']
        for err in errors:
            if err in runtime.stderr:
                self.raise_exception(runtime)
        runtime.returncode = 0
        return runtime

    def _write_tcl_script(self):
        fid = open('snapshots.tcl', 'w')
        script = ['save_tiff $env(_SNAPSHOT_STEM)-lat.tif', 'make_lateral_view', 'rotate_brain_y 180', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-med.tif', 'make_lateral_view', 'rotate_brain_x 90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-ven.tif', 'make_lateral_view', 'rotate_brain_x -90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-dor.tif']
        if isdefined(self.inputs.six_images) and self.inputs.six_images:
            script.extend(['make_lateral_view', 'rotate_brain_y 90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-pos.tif', 'make_lateral_view', 'rotate_brain_y -90', 'redraw', 'save_tiff $env(_SNAPSHOT_STEM)-ant.tif'])
        script.append('exit')
        fid.write('\n'.join(script))
        fid.close()

    def _list_outputs(self):
        outputs = self._outputs().get()
        if not isdefined(self.inputs.screenshot_stem):
            stem = '%s_%s_%s' % (self.inputs.subject_id, self.inputs.hemi, self.inputs.surface)
        else:
            stem = self.inputs.screenshot_stem
            stem_args = self.inputs.stem_template_args
            if isdefined(stem_args):
                args = tuple([getattr(self.inputs, arg) for arg in stem_args])
                stem = stem % args
        snapshots = ['%s-lat.tif', '%s-med.tif', '%s-dor.tif', '%s-ven.tif']
        if self.inputs.six_images:
            snapshots.extend(['%s-pos.tif', '%s-ant.tif'])
        snapshots = [self._gen_fname(f % stem, suffix='') for f in snapshots]
        outputs['snapshots'] = snapshots
        return outputs

    def _gen_filename(self, name):
        if name == 'tcl_script':
            return 'snapshots.tcl'
        return None