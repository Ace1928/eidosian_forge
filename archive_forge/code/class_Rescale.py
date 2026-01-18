from ..utils.filemanip import fname_presuffix
from .base import SimpleInterface, TraitedSpec, BaseInterfaceInputSpec, traits, File
from .. import LooseVersion
class Rescale(SimpleInterface):
    """Rescale an image

    Rescales the non-zero portion of ``in_file`` to match the bounds of the
    non-zero portion of ``ref_file``.
    Reference values in the input and reference images are defined by the
    ``percentile`` parameter, and the reference values in each image are
    identified and the remaining values are scaled accordingly.
    In the case of ``percentile == 0``, the reference values are the maxima
    and minima of each image.
    If the ``invert`` parameter is set, the input file is inverted prior to
    rescaling.

    Examples
    --------

    To use a high-resolution T1w image as a registration target for a T2\\*
    image, it may be useful to invert the T1w image and rescale to the T2\\*
    range.
    Using the 1st and 99th percentiles may reduce the impact of outlier
    voxels.

    >>> from nipype.interfaces.image import Rescale
    >>> invert_t1w = Rescale(invert=True)
    >>> invert_t1w.inputs.in_file = 'structural.nii'
    >>> invert_t1w.inputs.ref_file = 'functional.nii'
    >>> invert_t1w.inputs.percentile = 1.
    >>> res = invert_t1w.run()  # doctest: +SKIP

    """
    input_spec = RescaleInputSpec
    output_spec = RescaleOutputSpec

    def _run_interface(self, runtime):
        import numpy as np
        import nibabel as nb
        img = nb.load(self.inputs.in_file)
        data = img.get_fdata()
        ref_data = nb.load(self.inputs.ref_file).get_fdata()
        in_mask = data > 0
        ref_mask = ref_data > 0
        q = [self.inputs.percentile, 100.0 - self.inputs.percentile]
        in_low, in_high = np.percentile(data[in_mask], q)
        ref_low, ref_high = np.percentile(ref_data[ref_mask], q)
        scale_factor = (ref_high - ref_low) / (in_high - in_low)
        signal = in_high - data if self.inputs.invert else data - in_low
        out_data = in_mask * (signal * scale_factor + ref_low)
        suffix = '_inv' if self.inputs.invert else '_rescaled'
        out_file = fname_presuffix(self.inputs.in_file, suffix=suffix, newpath=runtime.cwd)
        img.__class__(out_data, img.affine, img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime