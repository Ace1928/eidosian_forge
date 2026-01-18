import pythran.config as cfg
from collections import defaultdict
import os.path
import os
class PythranBuildExtMeta(type):

    def __getitem__(self, base):

        class PythranBuildExt(PythranBuildExtMixIn, base):
            pass
        return PythranBuildExt