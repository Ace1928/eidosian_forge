from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as spa
from ..external_grey_box import ExternalGreyBoxModel, ExternalGreyBoxBlock
class PressureDropTwoEqualitiesTwoOutputsScaleOutputs(PressureDropTwoEqualitiesTwoOutputs):

    def get_output_constraint_scaling_factors(self):
        return np.asarray([4.1, 4.2])