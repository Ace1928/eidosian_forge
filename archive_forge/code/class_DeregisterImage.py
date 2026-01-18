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
class DeregisterImage:

    @staticmethod
    def do_check_mode(module, connection, image_id):
        image = get_image_by_id(connection, image_id)
        if image is None:
            module.exit_json(changed=False)
        if 'ImageId' in image:
            module.exit_json(changed=True, msg='Would have deregistered AMI if not in check mode.')
        else:
            module.exit_json(msg=f'Image {image_id} has already been deregistered.', changed=False)

    @staticmethod
    def defer_purge_snapshots(image):

        def purge_snapshots(connection):
            try:
                for mapping in image.get('BlockDeviceMappings') or []:
                    snapshot_id = mapping.get('Ebs', {}).get('SnapshotId')
                    if snapshot_id is None:
                        continue
                    connection.delete_snapshot(aws_retry=True, SnapshotId=snapshot_id)
                    yield snapshot_id
            except is_boto3_error_code('InvalidSnapshot.NotFound'):
                pass
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                raise Ec2AmiFailure('Failed to delete snapshot.', e)
        return purge_snapshots

    @staticmethod
    def timeout(connection, image_id, wait_timeout):
        image = get_image_by_id(connection, image_id)
        wait_till = time.time() + wait_timeout
        while wait_till > time.time() and image is not None:
            image = get_image_by_id(connection, image_id)
            time.sleep(3)
        if wait_till <= time.time():
            raise Ec2AmiFailure('Timed out waiting for image to be deregistered.')

    @classmethod
    def do(cls, module, connection, image_id):
        """Entry point to deregister an image"""
        delete_snapshot = module.params.get('delete_snapshot')
        wait = module.params.get('wait')
        wait_timeout = module.params.get('wait_timeout')
        image = get_image_by_id(connection, image_id)
        if image is None:
            module.exit_json(changed=False)
        purge_snapshots = cls.defer_purge_snapshots(image)
        if 'ImageId' in image:
            try:
                connection.deregister_image(aws_retry=True, ImageId=image_id)
            except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                raise Ec2AmiFailure('Error deregistering image', e)
        else:
            module.exit_json(msg=f'Image {image_id} has already been deregistered.', changed=False)
        if wait:
            cls.timeout(connection, image_id, wait_timeout)
        exit_params = {'msg': 'AMI deregister operation complete.', 'changed': True}
        if delete_snapshot:
            exit_params['snapshots_deleted'] = list(purge_snapshots(connection))
        module.exit_json(**exit_params)