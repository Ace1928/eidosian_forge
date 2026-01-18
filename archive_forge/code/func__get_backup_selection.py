from typing import Union
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
def _get_backup_selection(client, module, plan_id, selection_id):
    try:
        result = client.get_backup_selection(BackupPlanId=plan_id, SelectionId=selection_id)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        module.fail_json_aws(e, msg=f'Failed to describe selection {selection_id}')
    return result or []