import re
import copy
import time
import base64
import warnings
from typing import List
from libcloud.pricing import get_size_price
from libcloud.utils.py3 import ET, b, basestring, ensure_string
from libcloud.utils.xml import findall, findattr, findtext, fixxpath
from libcloud.common.aws import DEFAULT_SIGNATURE_VERSION, AWSBaseResponse, SignedAWSConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.iso8601 import parse_date, parse_date_allow_empty
from libcloud.utils.publickey import get_pubkey_comment, get_pubkey_ssh2_fingerprint
from libcloud.compute.providers import Provider
from libcloud.compute.constants.ec2_region_details_partial import (
class EC2Response(AWSBaseResponse):
    """
    EC2 specific response parsing and error handling.
    """

    def parse_error(self):
        err_list = []
        msg = 'Failure: 403 Forbidden'
        if self.status == 403 and self.body[:len(msg)] == msg:
            raise InvalidCredsError(msg)
        try:
            body = ET.XML(self.body)
        except Exception:
            raise MalformedResponseError('Failed to parse XML', body=self.body, driver=EC2NodeDriver)
        for err in body.findall('Errors/Error'):
            code, message = list(err)
            err_list.append('{}: {}'.format(code.text, message.text))
            if code.text == 'InvalidClientTokenId':
                raise InvalidCredsError(err_list[-1])
            if code.text == 'SignatureDoesNotMatch':
                raise InvalidCredsError(err_list[-1])
            if code.text == 'AuthFailure':
                raise InvalidCredsError(err_list[-1])
            if code.text == 'OptInRequired':
                raise InvalidCredsError(err_list[-1])
            if code.text == 'IdempotentParameterMismatch':
                raise IdempotentParamError(err_list[-1])
            if code.text == 'InvalidKeyPair.NotFound':
                match = re.match(".*\\'(.+?)\\'.*", message.text)
                if match:
                    name = match.groups()[0]
                else:
                    name = None
                raise KeyPairDoesNotExistError(name=name, driver=self.connection.driver)
        return '\n'.join(err_list)