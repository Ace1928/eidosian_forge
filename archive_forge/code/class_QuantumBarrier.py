import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumBarrier(QuantumInstruction):
    """
    quantumBarrier
        : 'barrier' indexIdentifierList
    """

    def __init__(self, indexIdentifierList: List[Identifier]):
        self.indexIdentifierList = indexIdentifierList