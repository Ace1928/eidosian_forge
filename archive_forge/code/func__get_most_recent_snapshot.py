import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _get_most_recent_snapshot(snapshots, max_snapshot_age_secs=None, now=None):
    """
    Gets the most recently created snapshot and optionally filters the result
    if the snapshot is too old
    :param snapshots: list of snapshots to search
    :param max_snapshot_age_secs: filter the result if its older than this
    :param now: simulate time -- used for unit testing
    :return:
    """
    if len(snapshots) == 0:
        return None
    if not now:
        now = datetime.datetime.now(datetime.timezone.utc)
    youngest_snapshot = max(snapshots, key=lambda s: s['StartTime'])
    snapshot_start = youngest_snapshot['StartTime']
    snapshot_age = now - snapshot_start
    if max_snapshot_age_secs is not None:
        if snapshot_age.total_seconds() > max_snapshot_age_secs:
            return None
    return youngest_snapshot