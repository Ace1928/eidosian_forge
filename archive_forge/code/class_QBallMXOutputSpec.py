import os
from ...utils.filemanip import split_filename
from ..base import (
class QBallMXOutputSpec(TraitedSpec):
    qmat = File(exists=True, desc='Q-Ball reconstruction matrix')