import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumMeasurement(ASTNode):
    """
    quantumMeasurement
        : 'measure' indexIdentifierList
    """

    def __init__(self, identifierList: List[Identifier]):
        self.identifierList = identifierList