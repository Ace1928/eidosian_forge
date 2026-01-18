from nipype.interfaces.base import (
import os
class RobustStatisticsSegmenterOutputSpec(TraitedSpec):
    segmentedImageFileName = File(position=-1, desc='Segmented image', exists=True)