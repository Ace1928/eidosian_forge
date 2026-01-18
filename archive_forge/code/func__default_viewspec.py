import base64
import inspect
import json
import re
import urllib
from copy import deepcopy
from typing import List as LList
from .... import __version__ as wandb_ver
from .... import termlog, termwarn
from ....sdk.lib import ipython
from ...public import Api as PublicApi
from ...public import RetryingClient
from ._blocks import P, PanelGrid, UnknownBlock, WeaveBlock, block_mapping, weave_blocks
from .mutations import UPSERT_VIEW, VIEW_REPORT
from .runset import Runset
from .util import Attr, Base, Block, coalesce, generate_name, nested_get, nested_set
from .validators import OneOf, TypeValidator
@staticmethod
def _default_viewspec():
    return {'id': None, 'name': None, 'spec': {'version': 5, 'panelSettings': {}, 'blocks': [], 'width': 'readable', 'authors': [], 'discussionThreads': [], 'ref': {}}}