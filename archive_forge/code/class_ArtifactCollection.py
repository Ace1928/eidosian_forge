import json
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from wandb_gql import Client, gql
import wandb
from wandb.apis import public
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.errors.term import termlog
class ArtifactCollection:

    def __init__(self, client: Client, entity: str, project: str, name: str, type: str, attrs: Optional[Mapping[str, Any]]=None):
        self.client = client
        self.entity = entity
        self.project = project
        self.name = name
        self.type = type
        self._attrs = attrs
        if self._attrs is None:
            self.load()
        self._aliases = [a['node']['alias'] for a in self._attrs['aliases']['edges']]

    @property
    def id(self):
        return self._attrs['id']

    @normalize_exceptions
    def artifacts(self, per_page=50):
        """Artifacts."""
        return Artifacts(self.client, self.entity, self.project, self.name, self.type, per_page=per_page)

    @property
    def aliases(self):
        """Artifact Collection Aliases."""
        return self._aliases

    def load(self):
        query = gql('\n        query ArtifactCollection(\n            $entityName: String!,\n            $projectName: String!,\n            $artifactTypeName: String!,\n            $artifactCollectionName: String!,\n            $cursor: String,\n            $perPage: Int = 1000\n        ) {\n            project(name: $projectName, entityName: $entityName) {\n                artifactType(name: $artifactTypeName) {\n                    artifactCollection: %s(name: $artifactCollectionName) {\n                        id\n                        name\n                        description\n                        createdAt\n                        aliases(after: $cursor, first: $perPage){\n                            edges {\n                                node {\n                                    alias\n                                }\n                                cursor\n                            }\n                            pageInfo {\n                                endCursor\n                                hasNextPage\n                            }\n                        }\n                    }\n                }\n            }\n        }\n        ' % artifact_collection_edge_name(server_supports_artifact_collections_gql_edges(self.client)))
        response = self.client.execute(query, variable_values={'entityName': self.entity, 'projectName': self.project, 'artifactTypeName': self.type, 'artifactCollectionName': self.name})
        if response is None or response.get('project') is None or response['project'].get('artifactType') is None or (response['project']['artifactType'].get('artifactCollection') is None):
            raise ValueError('Could not find artifact type %s' % self.type)
        self._attrs = response['project']['artifactType']['artifactCollection']
        return self._attrs

    def change_type(self, new_type: str) -> None:
        """Change the type of the artifact collection.

        Arguments:
            new_type: The new collection type to use, freeform string.
        """
        if not self.is_sequence():
            raise ValueError('Artifact collection needs to be a sequence')
        termlog(f'Changing artifact collection type of {self.type} to {new_type}')
        template = '\n            mutation MoveArtifactCollection(\n                $artifactSequenceID: ID!\n                $destinationArtifactTypeName: String!\n            ) {\n                moveArtifactSequence(\n                input: {\n                    artifactSequenceID: $artifactSequenceID\n                    destinationArtifactTypeName: $destinationArtifactTypeName\n                }\n                ) {\n                artifactCollection {\n                    id\n                    name\n                    description\n                    __typename\n                }\n                }\n            }\n            '
        variable_values = {'artifactSequenceID': self.id, 'destinationArtifactTypeName': new_type}
        mutation = gql(template)
        self.client.execute(mutation, variable_values=variable_values)
        self.type = new_type

    @normalize_exceptions
    def is_sequence(self) -> bool:
        """Return True if this is a sequence."""
        query = gql('\n            query FindSequence($entity: String!, $project: String!, $collection: String!, $type: String!) {\n                project(name: $project, entityName: $entity) {\n                    artifactType(name: $type) {\n                        __typename\n                        artifactSequence(name: $collection) {\n                            __typename\n                        }\n                    }\n                }\n            }\n            ')
        variables = {'entity': self.entity, 'project': self.project, 'collection': self.name, 'type': self.type}
        res = self.client.execute(query, variable_values=variables)
        sequence = res['project']['artifactType']['artifactSequence']
        return sequence is not None and sequence['__typename'] == 'ArtifactSequence'

    @normalize_exceptions
    def delete(self):
        """Delete the entire artifact collection."""
        if self.is_sequence():
            mutation = gql('\n                mutation deleteArtifactSequence($id: ID!) {\n                    deleteArtifactSequence(input: {\n                        artifactSequenceID: $id\n                    }) {\n                        artifactCollection {\n                            state\n                        }\n                    }\n                }\n                ')
        else:
            mutation = gql('\n                mutation deleteArtifactPortfolio($id: ID!) {\n                    deleteArtifactPortfolio(input: {\n                        artifactPortfolioID: $id\n                    }) {\n                        artifactCollection {\n                            state\n                        }\n                    }\n                }\n                ')
        self.client.execute(mutation, variable_values={'id': self.id})

    def __repr__(self):
        return f'<ArtifactCollection {self.name} ({self.type})>'