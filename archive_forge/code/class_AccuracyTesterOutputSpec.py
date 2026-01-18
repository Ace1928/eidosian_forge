from ..base import (
import os
class AccuracyTesterOutputSpec(TraitedSpec):
    output_directory = Directory(desc='Path to folder in which to store the results of the accuracy test.', argstr='%s', position=1)