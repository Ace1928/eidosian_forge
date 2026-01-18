from subprocess import Popen, PIPE
from ase.calculators.calculator import Calculator
from ase.io import read
from .create_input import GenerateVaspInput
import time
import os
import sys
def _stdout(self, text):
    if self.txt is not None:
        self.txt.write(text)
    if self.print_log:
        print(text, end='')