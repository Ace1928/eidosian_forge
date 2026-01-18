import os
import re as regex
from ..base import (
class BfcOutputSpec(TraitedSpec):
    outputMRIVolume = File(desc='path/name of output file')
    outputBiasField = File(desc='path/name of bias field output file')
    outputMaskedBiasField = File(desc='path/name of masked bias field output')
    correctionScheduleFile = File(desc='path/name of schedule file')