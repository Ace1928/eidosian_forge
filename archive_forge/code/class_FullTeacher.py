from parlai.core.teachers import FbDialogTeacher
from .build_2009 import build as build_2009
from .build_2018 import build as build_2018
import copy
import os
class FullTeacher(FbDialogTeacher):
    """
    This version of opensubtitles creates all possible dialog examples.
    """

    def __init__(self, opt, shared=None, version='2018', use_history=True):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, version, use_history)
        if not opt['datatype'].startswith('train'):
            opt['cands_datafile'] = opt['datafile']
        super().__init__(opt, shared)

    def setup_data(self, path):

        def rebuild(entries):
            if len(entries) == 0:
                return []
            flipped = [(SILENCE_TOKEN, [entries[0][0]], 0)]
            flipped += [(entries[i][1][0], [entries[i + 1][0]], 0) for i in range(len(entries) - 1)]
            return flipped
        alternate = []
        for entry, new in super().setup_data(path):
            if new:
                for i, e in enumerate(rebuild(alternate)):
                    if e[1]:
                        yield (e, i == 0)
                alternate.clear()
            else:
                alternate.append(entry)
            if entry[1]:
                yield (entry, new)
        if alternate:
            for i, e in enumerate(rebuild(alternate)):
                if e[1]:
                    yield (e, i == 0)
            alternate.clear()