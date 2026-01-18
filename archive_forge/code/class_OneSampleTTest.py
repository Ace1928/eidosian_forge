import os
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import FSCommand, FSTraitedSpec
from .utils import copy2subjdir
class OneSampleTTest(GLMFit):

    def __init__(self, **kwargs):
        super(OneSampleTTest, self).__init__(**kwargs)
        self.inputs.one_sample = True