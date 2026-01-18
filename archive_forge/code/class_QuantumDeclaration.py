import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumDeclaration(ASTNode):
    """
    quantumDeclaration
        : 'qreg' Identifier designator? |   # NOT SUPPORTED
         'qubit' designator? Identifier
    """

    def __init__(self, identifier: Identifier, designator=None):
        self.identifier = identifier
        self.designator = designator