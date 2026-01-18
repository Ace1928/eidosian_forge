from time import sleep
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _wait_for_status(self, awaited_status):
    """Wait for status to change from present status to awaited_status"""
    status_map = {'creating': 'available', 'rebooting': 'available', 'modifying': 'available', 'deleting': 'gone'}
    if self.status == awaited_status:
        return
    if status_map[self.status] != awaited_status:
        self.module.fail_json(msg=f"Invalid awaited status. '{self.status}' cannot transition to '{awaited_status}'")
    if awaited_status not in set(status_map.values()):
        self.module.fail_json(msg=f"'{awaited_status}' is not a valid awaited status.")
    while True:
        sleep(1)
        self._refresh_data()
        if self.status == awaited_status:
            break