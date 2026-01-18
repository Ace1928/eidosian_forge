import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class ArtifactTypes(Paginator):
    QUERY = gql('\n        query ProjectArtifacts(\n            $entityName: String!,\n            $projectName: String!,\n            $cursor: String,\n        ) {\n            project(name: $projectName, entityName: $entityName) {\n                artifactTypes(after: $cursor) {\n                    ...ArtifactTypesFragment\n                }\n            }\n        }\n        %s\n    ' % ARTIFACTS_TYPES_FRAGMENT)

    def __init__(self, client: Client, entity: str, project: str, per_page: Optional[int]=50):
        self.entity = entity
        self.project = project
        variable_values = {'entityName': entity, 'projectName': project}
        super().__init__(client, variable_values, per_page)

    @property
    def length(self):
        return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response['project']['artifactTypes']['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['project']['artifactTypes']['edges'][-1]['cursor']
        else:
            return None

    def update_variables(self):
        self.variables.update({'cursor': self.cursor})

    def convert_objects(self):
        if self.last_response['project'] is None:
            return []
        return [ArtifactType(self.client, self.entity, self.project, r['node']['name'], r['node']) for r in self.last_response['project']['artifactTypes']['edges']]