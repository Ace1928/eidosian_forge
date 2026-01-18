import shlex
from collections.abc import Mapping
def _replace_dict_in_list(lst):
    ans = []
    for v in lst:
        if type(v) is dict:
            ans.append(Bunch())
            ans[-1].update(v)
        elif type(v) is list:
            ans.append(_replace_dict_in_list(v))
        else:
            ans.append(v)
    return ans