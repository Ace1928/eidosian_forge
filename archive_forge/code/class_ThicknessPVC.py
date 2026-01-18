import os
import re as regex
from ..base import (
class ThicknessPVC(CommandLine):
    """
    ThicknessPVC computes cortical thickness using partial tissue fractions.
    This thickness measure is then transferred to the atlas surface to
    facilitate population studies. It also stores the computed thickness into
    separate hemisphere files and subject thickness mapped to the atlas
    hemisphere surfaces. ThicknessPVC is not run through the main SVReg
    sequence, and should be used after executing the BrainSuite and SVReg
    sequence.
    For more informaction, please see:

    http://brainsuite.org/processing/svreg/svreg_modules/

    Examples
    --------

    >>> from nipype.interfaces import brainsuite
    >>> thicknessPVC = brainsuite.ThicknessPVC()
    >>> thicknessPVC.inputs.subjectFilePrefix = 'home/user/btestsubject/testsubject'
    >>> results = thicknessPVC.run() #doctest: +SKIP

    """
    input_spec = ThicknessPVCInputSpec
    _cmd = 'thicknessPVC.sh'