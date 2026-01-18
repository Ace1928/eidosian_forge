import inspect
import re
import urllib
from typing import List as LList
from typing import Optional, Union
from .... import __version__ as wandb_ver
from .... import termwarn
from ...public import Api as PublicApi
from ._panels import UnknownPanel, WeavePanel, panel_mapping, weave_panels
from .runset import Runset
from .util import (
from .validators import OneOf, TypeValidator
class WeaveBlockSummaryTable(Block):
    """This is a hacky solution to support the most common way of getting Weave tables for now..."""
    entity: str = Attr()
    project: str = Attr()
    table_name: str = Attr()

    def __init__(self, entity, project, table_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity = entity
        self.project = project
        self.table_name = table_name

    @classmethod
    def from_json(cls, spec: dict) -> 'WeaveBlockSummaryTable':
        entity = spec['config']['panelConfig']['exp']['fromOp']['inputs']['obj']['fromOp']['inputs']['run']['fromOp']['inputs']['project']['fromOp']['inputs']['entityName']['val']
        project = spec['config']['panelConfig']['exp']['fromOp']['inputs']['obj']['fromOp']['inputs']['run']['fromOp']['inputs']['project']['fromOp']['inputs']['projectName']['val']
        table_name = spec['config']['panelConfig']['exp']['fromOp']['inputs']['key']['val']
        return cls(entity, project, table_name)

    @property
    def spec(self) -> dict:
        return {'type': 'weave-panel', 'children': [{'text': ''}], 'config': {'panelConfig': {'exp': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project'}}}, 'value': {'type': 'list', 'objectType': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'run': 'run'}}, 'value': {'type': 'union', 'members': [{'type': 'file', 'extension': 'json', 'wbObjectType': {'type': 'table', 'columnTypes': {}}}, 'none']}}}}, 'fromOp': {'name': 'pick', 'inputs': {'obj': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project'}}}, 'value': {'type': 'list', 'objectType': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'run': 'run'}}, 'value': {'type': 'union', 'members': [{'type': 'typedDict', 'propertyTypes': {'_wandb': {'type': 'typedDict', 'propertyTypes': {'runtime': 'number'}}}}, {'type': 'typedDict', 'propertyTypes': {'_step': 'number', 'table': {'type': 'file', 'extension': 'json', 'wbObjectType': {'type': 'table', 'columnTypes': {}}}, '_wandb': {'type': 'typedDict', 'propertyTypes': {'runtime': 'number'}}, '_runtime': 'number', '_timestamp': 'number'}}, {'type': 'typedDict', 'propertyTypes': {}}]}}}}, 'fromOp': {'name': 'run-summary', 'inputs': {'run': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project'}}}, 'value': {'type': 'list', 'objectType': 'run'}}, 'fromOp': {'name': 'project-runs', 'inputs': {'project': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': 'project'}, 'fromOp': {'name': 'root-project', 'inputs': {'entityName': {'nodeType': 'const', 'type': 'string', 'val': self.entity}, 'projectName': {'nodeType': 'const', 'type': 'string', 'val': self.project}}}}}}}}}}, 'key': {'nodeType': 'const', 'type': 'string', 'val': self.table_name}}}, '__userInput': True}}}}