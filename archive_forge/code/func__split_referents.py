from parlai.core.teachers import FixedDialogTeacher
from .build import build
import os
def _split_referents(self, raw_referents, spans):
    """
        Split the referents.

        The first 3 values are begin idx, end idx, and eos idx The next N_OBJECT values
        are booleans of if the object is referred e.g. 3 4 10 0 1 0 0 0 0 0 means idx 3
        to 4 is a markable of an utterance with <eos> at idx 10, and it refers to the
        2nd dot
        """
    referent_len = 3 + N_OBJECT
    splitted_referents = []
    for i in range(len(raw_referents) // referent_len):
        val = raw_referents[i * referent_len:(i + 1) * referent_len]
        splitted_referents.append(list(map(int, val)))
    referents = []
    idx = 0
    for span in spans:
        if span is None:
            referents.append(None)
            continue
        refs = []
        while idx < len(splitted_referents):
            if splitted_referents[idx][REF_EOS_IDX] == span[1]:
                ref = {'begin': splitted_referents[idx][REF_BEGIN_IDX] - (span[0] + 1), 'end': splitted_referents[idx][REF_END_IDX] - (span[0] + 1), 'target': splitted_referents[idx][REF_BEGIN_TARGET_IDX:]}
                refs.append(ref)
                idx += 1
            else:
                break
        referents.append(refs)
    return referents