from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018
import copy
import os
class Task100kTeacher(HalfTeacher):
    """
    This version of opensubtitles only includes 100,000 dialogs.
    """

    def setup_data(self, path):
        cnt = 0
        for entry, new in super().setup_data(path):
            if len(entry) > 1 and entry[1]:
                yield (entry, new)
            cnt += 1
            if cnt >= 100000:
                break