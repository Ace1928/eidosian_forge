import numpy as np
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import NiftySegCommand
from ..niftyreg.base import get_custom_path
class BinaryStatsInput(StatsInput):
    """Input Spec for seg_stats Binary operations."""
    operation = traits.Enum('p', 'sa', 'ss', 'svp', 'al', 'd', 'ncc', 'nmi', 'Vl', 'Nl', mandatory=True, argstr='-%s', position=4, desc='Operation to perform:\n\n    * p - <float> - The <float>th percentile of all voxels intensity (float=[0,100])\n    * sa - <ax> - Average of all voxels\n    * ss - <ax> - Standard deviation of all voxels\n    * svp - <ax> - Volume of all probabilsitic voxels (sum(<in>) x <volume per voxel>)\n    * al - <in2> - Average value in <in> for each label in <in2>\n    * d - <in2> - Calculate the Dice score between all classes in <in> and <in2>\n    * ncc - <in2> - Normalized cross correlation between <in> and <in2>\n    * nmi - <in2> - Normalized Mutual Information between <in> and <in2>\n    * Vl - <csv> - Volume of each integer label <in>. Save to <csv>file.\n    * Nl - <csv> - Count of each label <in>. Save to <csv> file.\n\n')
    operand_file = File(exists=True, argstr='%s', mandatory=True, position=5, xor=['operand_value'], desc='second image to perform operation with')
    operand_value = traits.Float(argstr='%.8f', mandatory=True, position=5, xor=['operand_file'], desc='value to perform operation with')