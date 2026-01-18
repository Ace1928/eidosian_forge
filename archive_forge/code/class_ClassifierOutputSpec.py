from ..base import (
import os
class ClassifierOutputSpec(TraitedSpec):
    artifacts_list_file = File(desc='Text file listing which ICs are artifacts; can be the output from classification or can be created manually')