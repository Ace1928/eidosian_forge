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
def _handle_fields(self, node):
    result = getattr(node, self.AST_FIELDS.get(type(node)))
    if isinstance(result, list):
        return [self._handle_fields(node) for node in result]
    elif isinstance(result, str):
        return self._unconvert(result)
    return result