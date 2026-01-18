import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class Artifacts(Paginator):
    """An iterable collection of artifact versions associated with a project and optional filter.

    This is generally used indirectly via the `Api`.artifact_versions method.
    """

    def __init__(self, client: Client, entity: str, project: str, collection_name: str, type: str, filters: Optional[Mapping[str, Any]]=None, order: Optional[str]=None, per_page: int=50):
        self.entity = entity
        self.collection_name = collection_name
        self.type = type
        self.project = project
        self.filters = {'state': 'COMMITTED'} if filters is None else filters
        self.order = order
        variables = {'project': self.project, 'entity': self.entity, 'order': self.order, 'type': self.type, 'collection': self.collection_name, 'filters': json.dumps(self.filters)}
        self.QUERY = gql('\n            query Artifacts($project: String!, $entity: String!, $type: String!, $collection: String!, $cursor: String, $perPage: Int = 50, $order: String, $filters: JSONString) {{\n                project(name: $project, entityName: $entity) {{\n                    artifactType(name: $type) {{\n                        artifactCollection: {}(name: $collection) {{\n                            name\n                            artifacts(filters: $filters, after: $cursor, first: $perPage, order: $order) {{\n                                totalCount\n                                edges {{\n                                    node {{\n                                        ...ArtifactFragment\n                                    }}\n                                    version\n                                    cursor\n                                }}\n                                pageInfo {{\n                                    endCursor\n                                    hasNextPage\n                                }}\n                            }}\n                        }}\n                    }}\n                }}\n            }}\n            {}\n            '.format(artifact_collection_edge_name(server_supports_artifact_collections_gql_edges(client)), wandb.Artifact._get_gql_artifact_fragment()))
        super().__init__(client, variables, per_page)

    @property
    def length(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifactCollection']['artifacts']['totalCount']
        else:
            return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifactCollection']['artifacts']['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['project']['artifactType']['artifactCollection']['artifacts']['edges'][-1]['cursor']
        else:
            return None

    def convert_objects(self):
        if self.last_response['project']['artifactType']['artifactCollection'] is None:
            return []
        return [wandb.Artifact._from_attrs(self.entity, self.project, self.collection_name + ':' + a['version'], a['node'], self.client) for a in self.last_response['project']['artifactType']['artifactCollection']['artifacts']['edges']]