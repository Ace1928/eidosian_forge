from plotly.basedatatypes import BaseLayoutType as _BaseLayoutType
import copy as _copy
def _subplot_re_match(self, prop):
    return self._subplotid_prop_re.match(prop)