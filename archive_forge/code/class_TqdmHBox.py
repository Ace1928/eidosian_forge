import re
import sys
from html import escape
from weakref import proxy
from .std import tqdm as std_tqdm
class TqdmHBox(HBox):
    """`ipywidgets.HBox` with a pretty representation"""

    def _json_(self, pretty=None):
        pbar = getattr(self, 'pbar', None)
        if pbar is None:
            return {}
        d = pbar.format_dict
        if pretty is not None:
            d['ascii'] = not pretty
        return d

    def __repr__(self, pretty=False):
        pbar = getattr(self, 'pbar', None)
        if pbar is None:
            return super(TqdmHBox, self).__repr__()
        return pbar.format_meter(**self._json_(pretty))

    def _repr_pretty_(self, pp, *_, **__):
        pp.text(self.__repr__(True))