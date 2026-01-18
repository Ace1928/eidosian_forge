import os.path as op
from ...utils.filemanip import split_filename
from ..base import (
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class ComputeTDI(MRTrix3Base):
    """
    Use track data as a form of contrast for producing a high-resolution
    image.

    .. admonition:: References

      * For TDI or DEC TDI: Calamante, F.; Tournier, J.-D.; Jackson, G. D. &
        Connelly, A. Track-density imaging (TDI): Super-resolution white
        matter imaging using whole-brain track-density mapping. NeuroImage,
        2010, 53, 1233-1243

      * If using -contrast length and -stat_vox mean: Pannek, K.; Mathias,
        J. L.; Bigler, E. D.; Brown, G.; Taylor, J. D. & Rose, S. E. The
        average pathlength map: A diffusion MRI tractography-derived index
        for studying brain pathology. NeuroImage, 2011, 55, 133-141

      * If using -dixel option with TDI contrast only: Smith, R.E., Tournier,
        J-D., Calamante, F., Connelly, A. A novel paradigm for automated
        segmentation of very large whole-brain probabilistic tractography
        data sets. In proc. ISMRM, 2011, 19, 673

      * If using -dixel option with any other contrast: Pannek, K., Raffelt,
        D., Salvado, O., Rose, S. Incorporating directional information in
        diffusion tractography derived maps: angular track imaging (ATI).
        In Proc. ISMRM, 2012, 20, 1912

      * If using -tod option: Dhollander, T., Emsell, L., Van Hecke, W., Maes,
        F., Sunaert, S., Suetens, P. Track Orientation Density Imaging (TODI)
        and Track Orientation Distribution (TOD) based tractography.
        NeuroImage, 2014, 94, 312-336

      * If using other contrasts / statistics: Calamante, F.; Tournier, J.-D.;
        Smith, R. E. & Connelly, A. A generalised framework for
        super-resolution track-weighted imaging. NeuroImage, 2012, 59,
        2494-2503

      * If using -precise mapping option: Smith, R. E.; Tournier, J.-D.;
        Calamante, F. & Connelly, A. SIFT: Spherical-deconvolution informed
        filtering of tractograms. NeuroImage, 2013, 67, 298-312 (Appendix 3)



    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tdi = mrt.ComputeTDI()
    >>> tdi.inputs.in_file = 'dti.mif'
    >>> tdi.cmdline                               # doctest: +ELLIPSIS
    'tckmap dti.mif tdi.mif'
    >>> tdi.run()                                 # doctest: +SKIP
    """
    _cmd = 'tckmap'
    input_spec = ComputeTDIInputSpec
    output_spec = ComputeTDIOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs