import os.path as op
import nibabel as nb
import numpy as np
from nibabel.volumeutils import native_code
from nibabel.orientations import aff2axcodes
from ... import logging
from ...utils.filemanip import split_filename
from ..base import TraitedSpec, File, isdefined
from ..dipy.base import DipyBaseInterface, HAVE_DIPY as have_dipy
class MRTrix2TrackVis(DipyBaseInterface):
    """
    Converts MRtrix (.tck) tract files into TrackVis (.trk) format
    using functions from dipy
    Example
    -------
    >>> import nipype.interfaces.mrtrix as mrt
    >>> tck2trk = mrt.MRTrix2TrackVis()
    >>> tck2trk.inputs.in_file = 'dwi_CSD_tracked.tck'
    >>> tck2trk.inputs.image_file = 'diffusion.nii'
    >>> tck2trk.run()                                   # doctest: +SKIP
    """
    input_spec = MRTrix2TrackVisInputSpec
    output_spec = MRTrix2TrackVisOutputSpec

    def _run_interface(self, runtime):
        from dipy.tracking.utils import affine_from_fsl_mat_file
        try:
            from dipy.tracking.utils import transform_tracking_output
        except ImportError:
            from dipy.tracking.utils import move_streamlines as transform_tracking_output
        dx, dy, dz = get_data_dims(self.inputs.image_file)
        vx, vy, vz = get_vox_dims(self.inputs.image_file)
        image_file = nb.load(self.inputs.image_file)
        affine = image_file.affine
        out_filename = op.abspath(self.inputs.out_filename)
        header, streamlines = read_mrtrix_tracks(self.inputs.in_file, as_generator=True)
        iflogger.info('MRTrix Header:')
        iflogger.info(header)
        trk_header = nb.trackvis.empty_header()
        trk_header['dim'] = [dx, dy, dz]
        trk_header['voxel_size'] = [vx, vy, vz]
        trk_header['n_count'] = header['count']
        if isdefined(self.inputs.matrix_file) and isdefined(self.inputs.registration_image_file):
            iflogger.info('Applying transformation from matrix file %s', self.inputs.matrix_file)
            xfm = np.genfromtxt(self.inputs.matrix_file)
            iflogger.info(xfm)
            registration_image_file = nb.load(self.inputs.registration_image_file)
            reg_affine = registration_image_file.affine
            r_dx, r_dy, r_dz = get_data_dims(self.inputs.registration_image_file)
            r_vx, r_vy, r_vz = get_vox_dims(self.inputs.registration_image_file)
            iflogger.info('Using affine from registration image file %s', self.inputs.registration_image_file)
            iflogger.info(reg_affine)
            trk_header['vox_to_ras'] = reg_affine
            trk_header['dim'] = [r_dx, r_dy, r_dz]
            trk_header['voxel_size'] = [r_vx, r_vy, r_vz]
            affine = np.dot(affine, np.diag(1.0 / np.array([vx, vy, vz, 1])))
            transformed_streamlines = transform_to_affine(streamlines, trk_header, affine)
            aff = affine_from_fsl_mat_file(xfm, [vx, vy, vz], [r_vx, r_vy, r_vz])
            iflogger.info(aff)
            axcode = aff2axcodes(reg_affine)
            trk_header['voxel_order'] = axcode[0] + axcode[1] + axcode[2]
            final_streamlines = transform_tracking_output(transformed_streamlines, aff)
            trk_tracks = ((ii, None, None) for ii in final_streamlines)
            nb.trackvis.write(out_filename, trk_tracks, trk_header)
            iflogger.info('Saving transformed Trackvis file as %s', out_filename)
            iflogger.info('New TrackVis Header:')
            iflogger.info(trk_header)
        else:
            iflogger.info('Applying transformation from scanner coordinates to %s', self.inputs.image_file)
            axcode = aff2axcodes(affine)
            trk_header['voxel_order'] = axcode[0] + axcode[1] + axcode[2]
            trk_header['vox_to_ras'] = affine
            transformed_streamlines = transform_to_affine(streamlines, trk_header, affine)
            trk_tracks = ((ii, None, None) for ii in transformed_streamlines)
            nb.trackvis.write(out_filename, trk_tracks, trk_header)
            iflogger.info('Saving Trackvis file as %s', out_filename)
            iflogger.info('TrackVis Header:')
            iflogger.info(trk_header)
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self.inputs.out_filename)
        return outputs

    def _gen_filename(self, name):
        if name == 'out_filename':
            return self._gen_outfilename()
        else:
            return None

    def _gen_outfilename(self):
        _, name, _ = split_filename(self.inputs.in_file)
        return name + '.trk'