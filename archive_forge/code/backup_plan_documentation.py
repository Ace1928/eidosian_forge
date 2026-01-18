import json
from datetime import datetime
from typing import Optional
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.backup import get_plan_details
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters

    Deletes a Backup Plan

    module : AnsibleAWSModule object
    client : boto3 backup client connection object
    backup_plan_id : ID (*not* name or ARN) of Backup plan to delete
    