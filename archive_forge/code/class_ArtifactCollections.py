import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class ArtifactCollections(Paginator):

    def __init__(self, client: Client, entity: str, project: str, type_name: str, per_page: Optional[int]=50):
        self.entity = entity
        self.project = project
        self.type_name = type_name
        variable_values = {'entityName': entity, 'projectName': project, 'artifactTypeName': type_name}
        self.QUERY = gql('\n            query ProjectArtifactCollections(\n                $entityName: String!,\n                $projectName: String!,\n                $artifactTypeName: String!\n                $cursor: String,\n            ) {\n                project(name: $projectName, entityName: $entityName) {\n                    artifactType(name: $artifactTypeName) {\n                        artifactCollections: %s(after: $cursor) {\n                            pageInfo {\n                                endCursor\n                                hasNextPage\n                            }\n                            totalCount\n                            edges {\n                                node {\n                                    id\n                                    name\n                                    description\n                                    createdAt\n                                }\n                                cursor\n                            }\n                        }\n                    }\n                }\n            }\n        ' % artifact_collection_plural_edge_name(server_supports_artifact_collections_gql_edges(client)))
        super().__init__(client, variable_values, per_page)

    @property
    def length(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifactCollections']['totalCount']
        else:
            return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifactCollections']['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifactCollections']['edges'][-1]['cursor']
        else:
            return None

    def update_variables(self):
        self.variables.update({'cursor': self.cursor})

    def convert_objects(self):
        return [ArtifactCollection(self.client, self.entity, self.project, r['node']['name'], self.type_name) for r in self.last_response['project']['artifactType']['artifactCollections']['edges']]