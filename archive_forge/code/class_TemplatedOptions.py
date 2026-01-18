from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
class TemplatedOptions:
    TEMPLATABLE_OPTIONS = ('access_key', 'secret_key', 'session_token', 'profile', 'iam_role_name')

    def __init__(self, templar, options):
        self.original_options = options
        self.templar = templar

    def __getitem__(self, *args):
        return self.original_options.__getitem__(self, *args)

    def __setitem__(self, *args):
        return self.original_options.__setitem__(self, *args)

    def get(self, *args):
        value = self.original_options.get(*args)
        if not value:
            return value
        if args[0] not in self.TEMPLATABLE_OPTIONS:
            return value
        if not self.templar.is_template(value):
            return value
        return self.templar.template(variable=value, disable_lookups=False)