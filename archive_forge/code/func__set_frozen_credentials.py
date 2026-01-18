from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def _set_frozen_credentials(self):
    options = self.get_options()
    iam_role_arn = options.get('assume_role_arn')
    if iam_role_arn:
        self._freeze_iam_role(iam_role_arn)