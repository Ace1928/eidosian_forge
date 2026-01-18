import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
def _replace_numeric_dots(self, s):
    numeric_dots = []
    for i, (left, mid, right) in enumerate(zip(s, s[1:], s[2:]), 1):
        if mid == '.':
            if left.isdigit() and right.isdigit() or (left.isdigit() and right == ' ') or (left == ' ' and right.isdigit()):
                numeric_dots.append(i)
    if s[-2].isdigit() and s[-1] == '.':
        numeric_dots.append(len(s) - 1)
    numeric_dots = [-1] + numeric_dots + [len(s)]
    substrs = []
    for start, stop in zip(numeric_dots, numeric_dots[1:]):
        substrs.append(s[start + 1:stop])
        substrs.append(self.DECIMAL_SPACER)
    substrs = substrs[:-1]
    return ''.join(substrs)