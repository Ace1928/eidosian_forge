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
class WeaveBlockArtifact(Block):
    """This is a hacky solution to support the most common way of getting Weave artifacts for now..."""
    entity: str = Attr()
    project: str = Attr()
    artifact: str = Attr()
    tab: str = Attr(validators=[OneOf(['overview', 'metadata', 'usage', 'files', 'lineage'])])

    def __init__(self, entity, project, artifact, tab='overview', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity = entity
        self.project = project
        self.artifact = artifact
        self.tab = tab

    @classmethod
    def from_json(cls, spec: dict) -> 'WeaveBlockSummaryTable':
        inputs = weave_inputs(spec)
        entity = inputs['project']['fromOp']['inputs']['entityName']['val']
        project = inputs['project']['fromOp']['inputs']['projectName']['val']
        artifact = inputs['artifactName']['val']
        tab = spec['config']['panelConfig']['panelConfig']['tabConfigs']['overview']['selectedTab']
        return cls(entity, project, artifact, tab)

    @property
    def spec(self) -> dict:
        return {'type': 'weave-panel', 'children': [{'text': ''}], 'config': {'panelConfig': {'exp': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string'}}}, 'value': 'artifact'}, 'fromOp': {'name': 'project-artifact', 'inputs': {'project': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': 'project'}, 'fromOp': {'name': 'root-project', 'inputs': {'entityName': {'nodeType': 'const', 'type': 'string', 'val': self.entity}, 'projectName': {'nodeType': 'const', 'type': 'string', 'val': self.project}}}}, 'artifactName': {'nodeType': 'const', 'type': 'string', 'val': self.artifact}}}, '__userInput': True}, 'panelInputType': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string'}}}, 'value': 'artifact'}, 'panelConfig': {'tabConfigs': {'overview': {'selectedTab': self.tab}}}}}}