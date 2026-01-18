import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumInstruction(ASTNode):
    """
    quantumInstruction
        : quantumGateCall
        | quantumPhase
        | quantumMeasurement
        | quantumReset
        | quantumBarrier
    """