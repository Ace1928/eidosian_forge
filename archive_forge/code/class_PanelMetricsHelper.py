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
class PanelMetricsHelper:
    FRONTEND_NAME_MAPPING = {'Step': '_step', 'Relative Time (Wall)': '_absolute_runtime', 'Relative Time (Process)': '_runtime', 'Wall Time': '_timestamp'}
    FRONTEND_NAME_MAPPING_REVERSED = {v: k for k, v in FRONTEND_NAME_MAPPING.items()}
    RUN_MAPPING = {'Created Timestamp': 'createdAt', 'Latest Timestamp': 'heartbeatAt'}
    RUN_MAPPING_REVERSED = {v: k for k, v in RUN_MAPPING.items()}

    def front_to_back(self, name):
        if name in self.FRONTEND_NAME_MAPPING:
            return self.FRONTEND_NAME_MAPPING[name]
        return name

    def back_to_front(self, name):
        if name in self.FRONTEND_NAME_MAPPING_REVERSED:
            return self.FRONTEND_NAME_MAPPING_REVERSED[name]
        return name

    def special_front_to_back(self, name):
        if name is None:
            return name
        name, *rest = name.split('.')
        rest = '.' + '.'.join(rest) if rest else ''
        if name.startswith('c::'):
            name = name[3:]
            return f'config:{name}.value{rest}'
        if name.startswith('s::'):
            name = name[3:] + rest
            return f'summary:{name}'
        name = name + rest
        if name in self.RUN_MAPPING:
            return 'run:' + self.RUN_MAPPING[name]
        if name in self.FRONTEND_NAME_MAPPING:
            return 'summary:' + self.FRONTEND_NAME_MAPPING[name]
        if name == 'Index':
            return name
        return 'summary:' + name

    def special_back_to_front(self, name):
        if name is not None:
            kind, rest = name.split(':', 1)
            if kind == 'config':
                pieces = rest.split('.')
                if len(pieces) <= 1:
                    raise ValueError(f'Invalid name: {name}')
                elif len(pieces) == 2:
                    name = pieces[0]
                elif len(pieces) >= 3:
                    name = pieces[:1] + pieces[2:]
                    name = '.'.join(name)
                return f'c::{name}'
            elif kind == 'summary':
                name = rest
                return f's::{name}'
        if name is None:
            return name
        elif 'summary:' in name:
            name = name.replace('summary:', '')
            return self.FRONTEND_NAME_MAPPING_REVERSED.get(name, name)
        elif 'run:' in name:
            name = name.replace('run:', '')
            return self.RUN_MAPPING_REVERSED[name]
        return name