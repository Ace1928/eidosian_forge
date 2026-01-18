import os
from ...utils.filemanip import split_filename
from ..base import (
class TrackPICoInputSpec(TrackInputSpec):
    pdf = traits.Enum('bingham', 'watson', 'acg', argstr='-pdf %s', desc='Specifies the model for PICo parameters. The default is "bingham.')
    iterations = traits.Int(argstr='-iterations %d', units='NA', desc='Number of streamlines to generate at each seed point. The default is 5000.')