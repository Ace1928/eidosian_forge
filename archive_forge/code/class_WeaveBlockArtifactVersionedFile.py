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
class WeaveBlockArtifactVersionedFile(Block):
    """This is a hacky solution to support the most common way of getting Weave artifact verions for now..."""
    entity: str = Attr()
    project: str = Attr()
    artifact: str = Attr()
    version: str = Attr()
    file: str = Attr()

    def __init__(self, entity, project, artifact, version, file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entity = entity
        self.project = project
        self.artifact = artifact
        self.version = version
        self.file = file

    @classmethod
    def from_json(cls, spec: dict) -> 'WeaveBlockSummaryTable':
        inputs = weave_inputs(spec)
        entity = inputs['artifactVersion']['fromOp']['inputs']['project']['fromOp']['inputs']['entityName']['val']
        project = inputs['artifactVersion']['fromOp']['inputs']['project']['fromOp']['inputs']['projectName']['val']
        artifact = inputs['artifactVersion']['fromOp']['inputs']['artifactName']['val']
        version = inputs['artifactVersion']['fromOp']['inputs']['artifactVersionAlias']['val']
        file = inputs['path']['val']
        return cls(entity, project, artifact, version, file)

    @property
    def spec(self) -> dict:
        return {'type': 'weave-panel', 'children': [{'text': ''}], 'config': {'panelConfig': {'exp': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string', 'artifactVersionAlias': 'string'}}}, 'value': {'type': 'file', 'extension': 'json', 'wbObjectType': {'type': 'table', 'columnTypes': {}}}}, 'fromOp': {'name': 'artifactVersion-file', 'inputs': {'artifactVersion': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string', 'artifactVersionAlias': 'string'}}}, 'value': 'artifactVersion'}, 'fromOp': {'name': 'project-artifactVersion', 'inputs': {'project': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': 'project'}, 'fromOp': {'name': 'root-project', 'inputs': {'entityName': {'nodeType': 'const', 'type': 'string', 'val': self.entity}, 'projectName': {'nodeType': 'const', 'type': 'string', 'val': self.project}}}}, 'artifactName': {'nodeType': 'const', 'type': 'string', 'val': self.artifact}, 'artifactVersionAlias': {'nodeType': 'const', 'type': 'string', 'val': self.version}}}}, 'path': {'nodeType': 'const', 'type': 'string', 'val': self.file}}}, '__userInput': True}}}}