from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_message
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import ensure_ec2_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class AnsibleEc2Cagw:

    def __init__(self, module, results):
        self._module = module
        self._results = results
        self._connection = self._module.client('ec2', retry_decorator=AWSRetry.jittered_backoff())
        self._check_mode = self._module.check_mode

    def process(self):
        vpc_id = self._module.params.get('vpc_id')
        state = self._module.params.get('state', 'present')
        tags = self._module.params.get('tags')
        purge_tags = self._module.params.get('purge_tags')
        if state == 'present':
            self.ensure_cagw_present(vpc_id, tags, purge_tags)
        elif state == 'absent':
            self.ensure_cagw_absent(vpc_id)

    def get_matching_cagw(self, vpc_id, carrier_gateway_id=None):
        """
        Returns the carrier gateway found.
            Parameters:
                vpc_id (str): VPC ID
                carrier_gateway_id (str): Carrier Gateway ID, if specified
            Returns:
                cagw (dict): dict of cagw found, None if none found
        """
        filters = ansible_dict_to_boto3_filter_list({'vpc-id': vpc_id})
        try:
            if not carrier_gateway_id:
                cagws = describe_cagws_with_backoff(self._connection, Filters=filters)
            else:
                cagws = describe_cagws_with_backoff(self._connection, CarrierGatewayIds=[carrier_gateway_id])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e)
        cagw = None
        if len(cagws) > 1:
            self._module.fail_json(msg=f'EC2 returned more than one Carrier Gateway for VPC {vpc_id}, aborting')
        elif cagws:
            cagw = camel_dict_to_snake_dict(cagws[0])
        return cagw

    @staticmethod
    def get_cagw_info(cagw, vpc_id):
        return {'carrier_gateway_id': cagw['carrier_gateway_id'], 'tags': boto3_tag_list_to_ansible_dict(cagw['tags']), 'vpc_id': vpc_id}

    def ensure_cagw_absent(self, vpc_id):
        cagw = self.get_matching_cagw(vpc_id)
        if cagw is None:
            return self._results
        if self._check_mode:
            self._results['changed'] = True
            return self._results
        try:
            self._results['changed'] = True
            self._connection.delete_carrier_gateway(aws_retry=True, CarrierGatewayId=cagw['carrier_gateway_id'])
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            self._module.fail_json_aws(e, msg='Unable to delete Carrier Gateway')
        return self._results

    def ensure_cagw_present(self, vpc_id, tags, purge_tags):
        cagw = self.get_matching_cagw(vpc_id)
        if cagw is None:
            if self._check_mode:
                self._results['changed'] = True
                self._results['carrier_gateway_id'] = None
                return self._results
            try:
                response = self._connection.create_carrier_gateway(VpcId=vpc_id, aws_retry=True)
                cagw = camel_dict_to_snake_dict(response['CarrierGateway'])
                self._results['changed'] = True
            except is_boto3_error_message('You must be opted into a wavelength zone to create a carrier gateway.') as e:
                self._module.fail_json(msg='You must be opted into a wavelength zone to create a carrier gateway')
            except botocore.exceptions.WaiterError as e:
                self._module.fail_json_aws(e, msg='No Carrier Gateway exists.')
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                self._module.fail_json_aws(e, msg='Unable to create Carrier Gateway')
        self._results['changed'] |= ensure_ec2_tags(self._connection, self._module, cagw['carrier_gateway_id'], resource_type='carrier-gateway', tags=tags, purge_tags=purge_tags, retry_codes='InvalidCarrierGatewayID.NotFound')
        cagw = self.get_matching_cagw(vpc_id, carrier_gateway_id=cagw['carrier_gateway_id'])
        cagw_info = self.get_cagw_info(cagw, vpc_id)
        self._results.update(cagw_info)
        return self._results