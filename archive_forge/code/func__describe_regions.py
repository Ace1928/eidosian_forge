from ansible.plugins.inventory import BaseInventoryPlugin
from ansible.plugins.inventory import Cacheable
from ansible.plugins.inventory import Constructable
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.plugin_utils.base import AWSPluginBase
from ansible_collections.amazon.aws.plugins.plugin_utils.botocore import AnsibleBotocoreError
def _describe_regions(self, service):
    try:
        initial_region = self.region or 'us-east-1'
        client = self.client(service, region=initial_region)
        resp = client.describe_regions()
    except AttributeError:
        pass
    except is_boto3_error_code('UnauthorizedOperation'):
        self.warn(f'UnauthorizedOperation when trying to list {service} regions')
    except botocore.exceptions.NoRegionError:
        self.warn(f'NoRegionError when trying to list {service} regions')
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        self.warn(f'Unexpected error while trying to list {service} regions: {e}')
    else:
        regions = [x['RegionName'] for x in resp.get('Regions', [])]
        if regions:
            return regions
    return None