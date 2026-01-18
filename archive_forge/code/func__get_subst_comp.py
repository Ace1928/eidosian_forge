from collections import OrderedDict
from .. import Substance
def _get_subst_comp(rsys, odesys, comp_keys, skip_keys):
    subst_comp = []
    for subst_key in odesys.names:
        _d = OrderedDict()
        for k, v in rsys.substances[subst_key].composition.items():
            if k in skip_keys:
                continue
            _d[comp_keys.index(k)] = v
        subst_comp.append(_d)
    return subst_comp