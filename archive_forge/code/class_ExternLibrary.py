import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
class ExternLibrary(ABC):
    _name: str
    _path: str
    _symbols: Dict[str, Symbol]
    _format: bool
    _grouping: bool

    def __init__(self, name: str, path: str, format: bool=True, grouping: bool=True) -> None:
        """
        Abstract class for extern library.
        :param name: name of the library
        :param path: path of the library
        :param format: whether to format the generated stub file
        """
        self._name = name
        self._path = path
        self._symbols = {}
        self._format = format
        self._grouping = grouping

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def symbols(self) -> Dict[str, Symbol]:
        return self._symbols

    @property
    def grouping(self) -> bool:
        return self._grouping

    @abstractmethod
    def parse_symbols(self, input_file) -> None:
        pass

    @abstractmethod
    def _output_stubs(self) -> str:
        pass

    def generate_stub_file(self, output_dir) -> None:
        file_str = self._output_stubs()
        if file_str is None or len(file_str) == 0:
            raise Exception('file_str is empty')
        output_file = f'{output_dir}/{self._name}.py'
        with open(output_file, 'w') as f:
            f.write(file_str)
            f.close()
            if self._format:
                subprocess.Popen(['autopep8', '-a', '-r', '-i', output_file], stdout=subprocess.PIPE).communicate()
                subprocess.Popen(['isort', output_file], stdout=subprocess.PIPE).communicate()