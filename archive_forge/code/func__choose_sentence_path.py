from parlai.core.teachers import FbDialogTeacher, FixedDialogTeacher
from .build import build
import copy
import os
def _choose_sentence_path(opt):
    build(opt)
    datatype = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'rephrase_sentences', 'choose_sentence_' + datatype + '.txt')