import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
class Reports(Paginator):
    """Reports is an iterable collection of `BetaReport` objects."""
    QUERY = gql('\n        query ProjectViews($project: String!, $entity: String!, $reportCursor: String,\n            $reportLimit: Int!, $viewType: String = "runs", $viewName: String) {\n            project(name: $project, entityName: $entity) {\n                allViews(viewType: $viewType, viewName: $viewName, first:\n                    $reportLimit, after: $reportCursor) {\n                    edges {\n                        node {\n                            id\n                            name\n                            displayName\n                            description\n                            user {\n                                username\n                                photoUrl\n                            }\n                            spec\n                            updatedAt\n                        }\n                        cursor\n                    }\n                    pageInfo {\n                        endCursor\n                        hasNextPage\n                    }\n\n                }\n            }\n        }\n        ')

    def __init__(self, client, project, name=None, entity=None, per_page=50):
        self.project = project
        self.name = name
        variables = {'project': project.name, 'entity': project.entity, 'viewName': self.name}
        super().__init__(client, variables, per_page)

    @property
    def length(self):
        if self.last_response:
            return len(self.objects)
        else:
            return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response['project']['allViews']['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['project']['allViews']['edges'][-1]['cursor']
        else:
            return None

    def update_variables(self):
        self.variables.update({'reportCursor': self.cursor, 'reportLimit': self.per_page})

    def convert_objects(self):
        if self.last_response['project'] is None:
            raise ValueError(f'Project {self.variables['project']} does not exist under entity {self.variables['entity']}')
        return [BetaReport(self.client, r['node'], entity=self.project.entity, project=self.project.name) for r in self.last_response['project']['allViews']['edges']]

    def __repr__(self):
        return '<Reports {}>'.format('/'.join(self.project.path))