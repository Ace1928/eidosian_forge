import os
from .base import ANTSCommand, ANTSCommandInputSpec
from ..base import TraitedSpec, File, traits, isdefined, InputMultiObject
from ...utils.filemanip import split_filename
class ApplyTransformsToPoints(ANTSCommand):
    """ApplyTransformsToPoints, applied to an CSV file, transforms coordinates
    using provided transform (or a set of transforms).

    Examples
    --------

    >>> from nipype.interfaces.ants import ApplyTransforms
    >>> at = ApplyTransformsToPoints()
    >>> at.inputs.dimension = 3
    >>> at.inputs.input_file = 'moving.csv'
    >>> at.inputs.transforms = ['trans.mat', 'ants_Warp.nii.gz']
    >>> at.inputs.invert_transform_flags = [False, False]
    >>> at.cmdline
    'antsApplyTransformsToPoints --dimensionality 3 --input moving.csv --output moving_transformed.csv --transform [ trans.mat, 0 ] --transform [ ants_Warp.nii.gz, 0 ]'


    """
    _cmd = 'antsApplyTransformsToPoints'
    input_spec = ApplyTransformsToPointsInputSpec
    output_spec = ApplyTransformsToPointsOutputSpec

    def _get_transform_filenames(self):
        retval = []
        for ii in range(len(self.inputs.transforms)):
            if isdefined(self.inputs.invert_transform_flags):
                if len(self.inputs.transforms) == len(self.inputs.invert_transform_flags):
                    invert_code = 1 if self.inputs.invert_transform_flags[ii] else 0
                    retval.append('--transform [ %s, %d ]' % (self.inputs.transforms[ii], invert_code))
                else:
                    raise Exception('ERROR: The useInverse list must have the same number of entries as the transformsFileName list.')
            else:
                retval.append('--transform %s' % self.inputs.transforms[ii])
        return ' '.join(retval)

    def _format_arg(self, opt, spec, val):
        if opt == 'transforms':
            return self._get_transform_filenames()
        return super(ApplyTransformsToPoints, self)._format_arg(opt, spec, val)