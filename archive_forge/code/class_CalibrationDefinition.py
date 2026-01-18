import enum
from typing import Optional, List, Union, Iterable, Tuple
class CalibrationDefinition(Statement):
    """
    calibrationDefinition
        : 'defcal' Identifier
        ( LPAREN calibrationArgumentList? RPAREN )? identifierList
        returnSignature? LBRACE .*? RBRACE  // for now, match anything inside body
        ;
    """

    def __init__(self, name: Identifier, identifierList: List[Identifier], calibrationArgumentList: Optional[List[CalibrationArgument]]=None):
        self.name = name
        self.identifierList = identifierList
        self.calibrationArgumentList = calibrationArgumentList or []