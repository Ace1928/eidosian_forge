import json
import requests
from wandb_gql import gql
from wandb_gql.client import RetryError
from wandb import util
from wandb.apis.normalize import normalize_exceptions
from wandb.sdk.lib import retry
@normalize_exceptions
@retry.retriable(check_retry_fn=util.no_retry_auth, retryable_exceptions=(RetryError, requests.RequestException))
def _load_next(self):
    max_step = self.page_offset + self.page_size
    if max_step > self.max_step:
        max_step = self.max_step
    variables = {'entity': self.run.entity, 'project': self.run.project, 'run': self.run.id, 'spec': json.dumps({'keys': self.keys, 'minStep': int(self.page_offset), 'maxStep': int(max_step), 'samples': int(self.page_size)})}
    res = self.client.execute(self.QUERY, variable_values=variables)
    res = res['project']['run']['sampledHistory']
    self.rows = res[0]
    self.page_offset += self.page_size
    self.scan_offset = 0