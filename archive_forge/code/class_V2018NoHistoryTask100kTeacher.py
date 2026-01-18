from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018
import copy
import os
class V2018NoHistoryTask100kTeacher(Task100kTeacher):
    """
    Note, these versions only uses two-turns dialog.

    This is more efficient due to movie-based deduplication, compared to the regular
    v2018 dataset.
    """

    def __init__(self, opt, shared=None):
        super(V2018NoHistoryTask100kTeacher, self).__init__(opt, shared, '2018', False)