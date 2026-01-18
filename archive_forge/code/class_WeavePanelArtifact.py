from typing import Optional, Union
from .helpers import LineKey, PCColumn
from .util import Attr, Panel, coalesce, nested_get, nested_set
from .validators import (
class WeavePanelArtifact(Panel):
    artifact: Optional[str] = Attr(json_path='spec.config.panel2Config.exp.fromOp.inputs.artifactName.val')
    tab: str = Attr(json_path='spec.config.panel2Config.panelConfig.tabConfigs.overview.selectedTab', validators=[OneOf(['overview', 'metadata', 'usage', 'files', 'lineage'])])

    def __init__(self, artifact, tab='overview', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._spec['config'] = self._default_config()
        self.artifact = artifact
        self.tab = tab

    @classmethod
    def from_json(cls, spec):
        artifact = spec['config']['panel2Config']['exp']['fromOp']['inputs']['artifactName']['val']
        tab = spec['config']['panel2Config']['panelConfig']['tabConfigs']['overview']['selectedTab']
        return cls(artifact, tab)

    @property
    def view_type(self) -> str:
        return 'Weave'

    @staticmethod
    def _default_config():
        return {'panel2Config': {'exp': {'nodeType': 'output', 'type': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string'}}}, 'value': 'artifact'}, 'fromOp': {'name': 'project-artifact', 'inputs': {'project': {'nodeType': 'var', 'type': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': 'project'}, 'varName': 'project'}, 'artifactName': {'nodeType': 'const', 'type': 'string', 'val': ''}}}, '__userInput': True}, 'panelInputType': {'type': 'tagged', 'tag': {'type': 'tagged', 'tag': {'type': 'typedDict', 'propertyTypes': {'entityName': 'string', 'projectName': 'string'}}, 'value': {'type': 'typedDict', 'propertyTypes': {'project': 'project', 'artifactName': 'string'}}}, 'value': 'artifact'}, 'panelConfig': {'tabConfigs': {'overview': {'selectedTab': 'overview'}}}}}