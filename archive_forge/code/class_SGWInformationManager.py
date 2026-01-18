from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
class SGWInformationManager(object):

    def __init__(self, client, module):
        self.client = client
        self.module = module
        self.name = self.module.params.get('name')

    def fetch(self):
        gateways = self.list_gateways()
        for gateway in gateways:
            if self.module.params.get('gather_local_disks'):
                self.list_local_disks(gateway)
            if gateway['gateway_type'] == 'FILE_S3' and self.module.params.get('gather_file_shares'):
                self.list_gateway_file_shares(gateway)
            elif gateway['gateway_type'] == 'VTL' and self.module.params.get('gather_tapes'):
                self.list_gateway_vtl(gateway)
            elif gateway['gateway_type'] in ['CACHED', 'STORED'] and self.module.params.get('gather_volumes'):
                self.list_gateway_volumes(gateway)
        self.module.exit_json(gateways=gateways)
    '\n    List all storage gateways for the AWS endpoint.\n    '

    def list_gateways(self):
        try:
            paginator = self.client.get_paginator('list_gateways')
            response = paginator.paginate(PaginationConfig={'PageSize': 100}).build_full_result()
            gateways = []
            for gw in response['Gateways']:
                gateways.append(camel_dict_to_snake_dict(gw))
            return gateways
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg="Couldn't list storage gateways")
    '\n    Read file share objects from AWS API response.\n    Drop the gateway_arn attribute from response, as it will be duplicate with parent object.\n    '

    @staticmethod
    def _read_gateway_fileshare_response(fileshares, aws_reponse):
        for share in aws_reponse['FileShareInfoList']:
            share_obj = camel_dict_to_snake_dict(share)
            if 'gateway_arn' in share_obj:
                del share_obj['gateway_arn']
            fileshares.append(share_obj)
        return aws_reponse['NextMarker'] if 'NextMarker' in aws_reponse else None
    '\n    List file shares attached to AWS storage gateway when in S3 mode.\n    '

    def list_gateway_file_shares(self, gateway):
        try:
            response = self.client.list_file_shares(GatewayARN=gateway['gateway_arn'], Limit=100)
            gateway['file_shares'] = []
            marker = self._read_gateway_fileshare_response(gateway['file_shares'], response)
            while marker is not None:
                response = self.client.list_file_shares(GatewayARN=gateway['gateway_arn'], Marker=marker, Limit=100)
                marker = self._read_gateway_fileshare_response(gateway['file_shares'], response)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg="Couldn't list gateway file shares")
    '\n    List storage gateway local disks\n    '

    def list_local_disks(self, gateway):
        try:
            gateway['local_disks'] = [camel_dict_to_snake_dict(disk) for disk in self.client.list_local_disks(GatewayARN=gateway['gateway_arn'])['Disks']]
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg="Couldn't list storage gateway local disks")
    '\n    Read tape objects from AWS API response.\n    Drop the gateway_arn attribute from response, as it will be duplicate with parent object.\n    '

    @staticmethod
    def _read_gateway_tape_response(tapes, aws_response):
        for tape in aws_response['TapeInfos']:
            tape_obj = camel_dict_to_snake_dict(tape)
            if 'gateway_arn' in tape_obj:
                del tape_obj['gateway_arn']
            tapes.append(tape_obj)
        return aws_response['Marker'] if 'Marker' in aws_response else None
    '\n    List VTL & VTS attached to AWS storage gateway in VTL mode\n    '

    def list_gateway_vtl(self, gateway):
        try:
            response = self.client.list_tapes(Limit=100)
            gateway['tapes'] = []
            marker = self._read_gateway_tape_response(gateway['tapes'], response)
            while marker is not None:
                response = self.client.list_tapes(Marker=marker, Limit=100)
                marker = self._read_gateway_tape_response(gateway['tapes'], response)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg="Couldn't list storage gateway tapes")
    '\n    List volumes attached to AWS storage gateway in CACHED or STORAGE mode\n    '

    def list_gateway_volumes(self, gateway):
        try:
            paginator = self.client.get_paginator('list_volumes')
            response = paginator.paginate(GatewayARN=gateway['gateway_arn'], PaginationConfig={'PageSize': 100}).build_full_result()
            gateway['volumes'] = []
            for volume in response['VolumeInfos']:
                volume_obj = camel_dict_to_snake_dict(volume)
                if 'gateway_arn' in volume_obj:
                    del volume_obj['gateway_arn']
                if 'gateway_id' in volume_obj:
                    del volume_obj['gateway_id']
                gateway['volumes'].append(volume_obj)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e, msg="Couldn't list storage gateway volumes")