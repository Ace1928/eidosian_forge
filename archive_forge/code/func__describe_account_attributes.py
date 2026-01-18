from ansible.errors import AnsibleLookupError
from ansible.module_utils._text import to_native
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.plugin_utils.lookup import AWSLookupBase
def _describe_account_attributes(client, **params):
    return client.describe_account_attributes(aws_retry=True, **params)