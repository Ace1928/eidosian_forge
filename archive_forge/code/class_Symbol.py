import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
class Symbol:
    _name: str
    _op_name: str
    _ret_type: str
    _arg_names: List[str]
    _arg_types: List[str]

    def __init__(self, name: str, op_name: str, ret_type: str, arg_names: List[str], arg_types: List[str]) -> None:
        """
        A symbol is a function declaration.
        :param name: name of the symbol
        :param op_name: name of the operation
        :param ret_type: return type of the operation
        :param arg_names: names of the arguments
        :param arg_types: types of the arguments
        """
        self._name = name
        self._op_name = op_name
        self._ret_type = ret_type
        self._arg_names = list(arg_names)
        self._arg_types = list(arg_types)

    @property
    def name(self) -> str:
        return self._name

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def ret_type(self) -> str:
        return self._ret_type

    @property
    def arg_names(self) -> List[str]:
        return self._arg_names

    @property
    def arg_types(self) -> List[str]:
        return self._arg_types