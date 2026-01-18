import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumGateDefinition(Statement):
    """
    quantumGateDefinition
        : 'gate' quantumGateSignature quantumBlock
    """

    def __init__(self, quantumGateSignature: QuantumGateSignature, quantumBlock: QuantumBlock):
        self.quantumGateSignature = quantumGateSignature
        self.quantumBlock = quantumBlock