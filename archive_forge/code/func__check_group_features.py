import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
def _check_group_features(info, parsed):
    """Checks whether the reverse and fuzzy features of the group calls match
    the groups which they call.
    """
    call_refs = {}
    additional_groups = []
    for call, reverse, fuzzy in info.group_calls:
        key = (call.group, reverse, fuzzy)
        ref = call_refs.get(key)
        if ref is None:
            if call.group == 0:
                rev = bool(info.flags & REVERSE)
                fuz = isinstance(parsed, Fuzzy)
                if (rev, fuz) != (reverse, fuzzy):
                    additional_groups.append((CallRef(len(call_refs), parsed), reverse, fuzzy))
            else:
                def_info = info.defined_groups[call.group]
                group = def_info[0]
                if def_info[1:] != (reverse, fuzzy):
                    additional_groups.append((group, reverse, fuzzy))
            ref = len(call_refs)
            call_refs[key] = ref
        call.call_ref = ref
    info.call_refs = call_refs
    info.additional_groups = additional_groups