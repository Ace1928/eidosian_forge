import os
from ....base import (
class fibertrackOutputSpec(TraitedSpec):
    output_fiber_file = File(desc='The filename for the fiber file produced by the algorithm. This file must end in a .fib or .vtk extension for ITK spatial object and vtkPolyData formats respectively.', exists=True)