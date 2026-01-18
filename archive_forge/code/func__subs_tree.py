import abc
import collections
import collections.abc
import operator
import sys
import typing
def _subs_tree(self, tvars=None, args=None):
    if self is Annotated:
        return Annotated
    res = super()._subs_tree(tvars=tvars, args=args)
    if isinstance(res[1], tuple) and res[1][0] is Annotated:
        sub_tp = res[1][1]
        sub_annot = res[1][2]
        return (Annotated, sub_tp, sub_annot + res[2])
    return res