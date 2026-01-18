from wandb_gql import gql
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.normalize import normalize_exceptions
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
class Projects(Paginator):
    """An iterable collection of `Project` objects."""
    QUERY = gql('\n        query Projects($entity: String, $cursor: String, $perPage: Int = 50) {\n            models(entityName: $entity, after: $cursor, first: $perPage) {\n                edges {\n                    node {\n                        ...ProjectFragment\n                    }\n                    cursor\n                }\n                pageInfo {\n                    endCursor\n                    hasNextPage\n                }\n            }\n        }\n        %s\n        ' % PROJECT_FRAGMENT)

    def __init__(self, client, entity, per_page=50):
        self.client = client
        self.entity = entity
        variables = {'entity': self.entity}
        super().__init__(client, variables, per_page)

    @property
    def length(self):
        return None

    @property
    def more(self):
        if self.last_response:
            return self.last_response['models']['pageInfo']['hasNextPage']
        else:
            return True

    @property
    def cursor(self):
        if self.last_response:
            return self.last_response['models']['edges'][-1]['cursor']
        else:
            return None

    def convert_objects(self):
        return [Project(self.client, self.entity, p['node']['name'], p['node']) for p in self.last_response['models']['edges']]

    def __repr__(self):
        return f'<Projects {self.entity}>'