import numpy as np
from pennylane import math
from pennylane.operation import AnyWires, Operation, StatePrepBase
from pennylane.templates.state_preparations import BasisStatePreparation, MottonenStatePreparation
from pennylane.wires import Wires, WireError
class QubitStateVector(StatePrep):
    pass