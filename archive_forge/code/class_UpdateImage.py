import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import add_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_specifications
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
class UpdateImage:

    @staticmethod
    def set_launch_permission(connection, image, launch_permissions, check_mode):
        if launch_permissions is None:
            return False
        current_permissions = image['LaunchPermissions']
        current_users = set((permission['UserId'] for permission in current_permissions if 'UserId' in permission))
        desired_users = set((str(user_id) for user_id in launch_permissions.get('user_ids', [])))
        current_groups = set((permission['Group'] for permission in current_permissions if 'Group' in permission))
        desired_groups = set(launch_permissions.get('group_names', []))
        current_org_arns = set((permission['OrganizationArn'] for permission in current_permissions if 'OrganizationArn' in permission))
        desired_org_arns = set((str(org_arn) for org_arn in launch_permissions.get('org_arns', [])))
        current_org_unit_arns = set((permission['OrganizationalUnitArn'] for permission in current_permissions if 'OrganizationalUnitArn' in permission))
        desired_org_unit_arns = set((str(org_unit_arn) for org_unit_arn in launch_permissions.get('org_unit_arns', [])))
        to_add_users = desired_users - current_users
        to_remove_users = current_users - desired_users
        to_add_groups = desired_groups - current_groups
        to_remove_groups = current_groups - desired_groups
        to_add_org_arns = desired_org_arns - current_org_arns
        to_remove_org_arns = current_org_arns - desired_org_arns
        to_add_org_unit_arns = desired_org_unit_arns - current_org_unit_arns
        to_remove_org_unit_arns = current_org_unit_arns - desired_org_unit_arns
        to_add = [dict(Group=group) for group in sorted(to_add_groups)] + [dict(UserId=user_id) for user_id in sorted(to_add_users)] + [dict(OrganizationArn=org_arn) for org_arn in sorted(to_add_org_arns)] + [dict(OrganizationalUnitArn=org_unit_arn) for org_unit_arn in sorted(to_add_org_unit_arns)]
        to_remove = [dict(Group=group) for group in sorted(to_remove_groups)] + [dict(UserId=user_id) for user_id in sorted(to_remove_users)] + [dict(OrganizationArn=org_arn) for org_arn in sorted(to_remove_org_arns)] + [dict(OrganizationalUnitArn=org_unit_arn) for org_unit_arn in sorted(to_remove_org_unit_arns)]
        if not (to_add or to_remove):
            return False
        try:
            if not check_mode:
                connection.modify_image_attribute(aws_retry=True, ImageId=image['ImageId'], Attribute='launchPermission', LaunchPermission=dict(Add=to_add, Remove=to_remove))
            changed = True
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            raise Ec2AmiFailure(f'Error updating launch permissions of image {image['ImageId']}', e)
        return changed

    @staticmethod
    def set_tags(connection, module, image_id, tags, purge_tags):
        if not tags:
            return False
        return ensure_ec2_tags(connection, module, image_id, tags=tags, purge_tags=purge_tags)

    @staticmethod
    def set_description(connection, module, image, description):
        if not description:
            return False
        if description == image['Description']:
            return False
        try:
            if not module.check_mode:
                connection.modify_image_attribute(aws_retry=True, Attribute='Description', ImageId=image['ImageId'], Description={'Value': description})
            return True
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            raise Ec2AmiFailure(f'Error setting description for image {image['ImageId']}', e)

    @classmethod
    def do(cls, module, connection, image_id):
        """Entry point to update an image"""
        launch_permissions = module.params.get('launch_permissions')
        if launch_permissions:
            launch_permissions = {k: v for k, v in launch_permissions.items() if v is not None}
        image = get_image_by_id(connection, image_id)
        if image is None:
            raise Ec2AmiFailure(f'Image {image_id} does not exist')
        changed = False
        changed |= cls.set_launch_permission(connection, image, launch_permissions, module.check_mode)
        changed |= cls.set_tags(connection, module, image_id, module.params['tags'], module.params['purge_tags'])
        changed |= cls.set_description(connection, module, image, module.params['description'])
        if changed and module.check_mode:
            module.exit_json(changed=True, msg='Would have updated AMI if not in check mode.')
        elif changed:
            module.exit_json(msg='AMI updated.', changed=True, **get_ami_info(get_image_by_id(connection, image_id)))
        else:
            module.exit_json(msg='AMI not updated.', changed=False, **get_ami_info(image))