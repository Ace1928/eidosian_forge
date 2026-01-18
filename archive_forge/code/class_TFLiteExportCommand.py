import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from ...exporters import TasksManager
from ...exporters.tflite import QuantizationApproach
from ..base import BaseOptimumCLICommand
class TFLiteExportCommand(BaseOptimumCLICommand):

    def __init__(self, subparsers: Optional['_SubParsersAction'], args: Optional['Namespace']=None, command: Optional['CommandInfo']=None, from_defaults_factory: bool=False, parser: Optional['ArgumentParser']=None):
        super().__init__(subparsers, args, command=command, from_defaults_factory=from_defaults_factory, parser=parser)
        self.args_string = ' '.join(sys.argv[3:])

    @staticmethod
    def parse_args(parser: 'ArgumentParser'):
        return parse_args_tflite(parser)

    def run(self):
        full_command = f'python3 -m optimum.exporters.tflite {self.args_string}'
        subprocess.run(full_command, shell=True, check=True)