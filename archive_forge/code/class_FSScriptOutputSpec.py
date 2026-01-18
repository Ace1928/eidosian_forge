import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
class FSScriptOutputSpec(TraitedSpec):
    log_file = File('output.nipype', usedefault=True, exists=True, desc='The output log')