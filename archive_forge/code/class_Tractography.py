import os.path as op
from ..base import traits, TraitedSpec, File
from .base import MRTrix3BaseInputSpec, MRTrix3Base
class Tractography(MRTrix3Base):
    """
    Performs streamlines tractography after selecting the appropriate algorithm.

    References
    ----------

    .. [FACT] Mori, S.; Crain, B. J.; Chacko, V. P. & van Zijl,
      P. C. M. Three-dimensional tracking of axonal projections in the
      brain by magnetic resonance imaging. Annals of Neurology, 1999,
      45, 265-269

    .. [iFOD1] Tournier, J.-D.; Calamante, F. & Connelly, A. MRtrix:
      Diffusion tractography in crossing fiber regions. Int. J. Imaging
      Syst. Technol., 2012, 22, 53-66

    .. [iFOD2] Tournier, J.-D.; Calamante, F. & Connelly, A. Improved
      probabilistic streamlines tractography by 2nd order integration
      over fibre orientation distributions. Proceedings of the
      International Society for Magnetic Resonance in Medicine, 2010, 1670

    .. [Nulldist] Morris, D. M.; Embleton, K. V. & Parker, G. J.
      Probabilistic fibre tracking: Differentiation of connections from
      chance events. NeuroImage, 2008, 42, 1329-1339

    .. [Tensor_Det] Basser, P. J.; Pajevic, S.; Pierpaoli, C.; Duda, J.
      and Aldroubi, A. In vivo fiber tractography using DT-MRI data.
      Magnetic Resonance in Medicine, 2000, 44, 625-632

    .. [Tensor_Prob] Jones, D. Tractography Gone Wild: Probabilistic Fibre
      Tracking Using the Wild Bootstrap With Diffusion Tensor MRI. IEEE
      Transactions on Medical Imaging, 2008, 27, 1268-1274

    Example
    -------

    >>> import nipype.interfaces.mrtrix3 as mrt
    >>> tk = mrt.Tractography()
    >>> tk.inputs.in_file = 'fods.mif'
    >>> tk.inputs.roi_mask = 'mask.nii.gz'
    >>> tk.inputs.seed_sphere = (80, 100, 70, 10)
    >>> tk.cmdline                               # doctest: +ELLIPSIS
    'tckgen -algorithm iFOD2 -samples 4 -output_seeds out_seeds.nii.gz -mask mask.nii.gz -seed_sphere 80.000000,100.000000,70.000000,10.000000 fods.mif tracked.tck'
    >>> tk.run()                                 # doctest: +SKIP
    """
    _cmd = 'tckgen'
    input_spec = TractographyInputSpec
    output_spec = TractographyOutputSpec

    def _format_arg(self, name, trait_spec, value):
        if 'roi_' in name and isinstance(value, tuple):
            value = ['%f' % v for v in value]
            return trait_spec.argstr % ','.join(value)
        return super(Tractography, self)._format_arg(name, trait_spec, value)

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        return outputs