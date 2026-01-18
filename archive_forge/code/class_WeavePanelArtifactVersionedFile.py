from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
class WeavePanelArtifactVersionedFile(Panel):
    artifact: str = Attr(json_path='spec.config.panel2Config.exp.fromOp.inputs.artifactVersion.fromOp.inputs.artifactName.val')
    version: str = Attr(json_path='spec.config.panel2Config.exp.fromOp.inputs.artifactVersion.fromOp.inputs.artifactVersionAlias.val')
    file: str = Attr(json_path='spec.config.panel2Config.exp.fromOp.inputs.path.val')

    def __init__(self, artifact, version, file, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec['config'] = self._default_config()
        self.artifact = artifact
        self.version = version
        self.file = file

    @classmethod
    def from_json(cls, spec):
        artifact = spec['config']['panel2Config']['exp']['fromOp']['inputs']['artifactVersion']['fromOp']['inputs']['artifactName']['val']
        version = spec['config']['panel2Config']['exp']['fromOp']['inputs']['artifactVersion']['fromOp']['inputs']['artifactVersionAlias']['val']
        file = spec['config']['panel2Config']['exp']['fromOp']['inputs']['path']['val']
        return cls(artifact, version, file)

    @property
    def view_type(self) -> str:
        return 'Weave'

    @staticmethod
    def _default_config():
        return {'panel2Config': {'exp': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string', 'artifactVersionAlias': 'string'}}}, 'value': {'type': 'file', 'extension': 'json', 'wbObjectType': {'type': 'table', 'columnTypes': {}}}}, 'fromOp': {'name': 'artifactVersion-file', 'inputs': {'artifactVersion': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string', 'artifactVersionAlias': 'string'}}}, 'value': 'artifactVersion'}, 'fromOp': {'name': 'project-artifactVersion', 'inputs': {'project': {'nodeType': 'var', 'type': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': 'project'}, 'varName': 'project'}, 'artifactName': {'nodeType': 'const', 'type': 'string', 'val': ''}, 'artifactVersionAlias': {'nodeType': 'const', 'type': 'string', 'val': ''}}}}, 'path': {'nodeType': 'const', 'type': 'string', 'val': ''}}}, '__userInput': True}}}