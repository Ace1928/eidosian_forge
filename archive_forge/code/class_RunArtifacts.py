import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class RunArtifacts(Paginator):

    def __init__(self, client: Client, run: 'Run', mode='logged', per_page: Optional[int]=50):
        output_query = gql('\n            query RunOutputArtifacts(\n                $entity: String!, $project: String!, $runName: String!, $cursor: String, $perPage: Int,\n            ) {\n                project(name: $project, entityName: $entity) {\n                    run(name: $runName) {\n                        outputArtifacts(after: $cursor, first: $perPage) {\n                            totalCount\n                            edges {\n                                node {\n                                    ...ArtifactFragment\n                                }\n                                cursor\n                            }\n                            pageInfo {\n                                endCursor\n                                hasNextPage\n                            }\n                        }\n                    }\n                }\n            }\n            ' + wandb.Artifact._get_gql_artifact_fragment())
        input_query = gql('\n            query RunInputArtifacts(\n                $entity: String!, $project: String!, $runName: String!, $cursor: String, $perPage: Int,\n            ) {\n                project(name: $project, entityName: $entity) {\n                    run(name: $runName) {\n                        inputArtifacts(after: $cursor, first: $perPage) {\n                            totalCount\n                            edges {\n                                node {\n                                    ...ArtifactFragment\n                                }\n                                cursor\n                            }\n                            pageInfo {\n                                endCursor\n                                hasNextPage\n                            }\n                        }\n                    }\n                }\n            }\n            ' + wandb.Artifact._get_gql_artifact_fragment())
        self.run = run
        if mode == 'logged':
            self.run_key = 'outputArtifacts'
            self.QUERY = output_query
        elif mode == 'used':
            self.run_key = 'inputArtifacts'
            self.QUERY = input_query
        else:
            raise ValueError('mode must be logged or used')
        variable_values = {'entity': run.entity, 'project': run.project, 'runName': run.id}
        super().__init__(client, variable_values, per_page)

    @property
    def length(self):
        if self.last_response:
            return self.last_response['project']['run'][self.run_key]['totalCount']
        else:
            return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response['project']['run'][self.run_key]['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['project']['run'][self.run_key]['edges'][-1]['cursor']
        else:
            return None

    def convert_objects(self):
        return [wandb.Artifact._from_attrs(r['node']['artifactSequence']['project']['entityName'], r['node']['artifactSequence']['project']['name'], '{}:v{}'.format(r['node']['artifactSequence']['name'], r['node']['versionIndex']), r['node'], self.client) for r in self.last_response['project']['run'][self.run_key]['edges']]