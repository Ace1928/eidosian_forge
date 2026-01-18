import base64
import datetime
import json
import weakref
import botocore
import botocore.auth
from botocore.awsrequest import create_request_object, prepare_request_dict
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import ArnParser, datetime2timestamp
from botocore.utils import fix_s3_host  # noqa
class CloudFrontSigner:
    """A signer to create a signed CloudFront URL.

    First you create a cloudfront signer based on a normalized RSA signer::

        import rsa
        def rsa_signer(message):
            private_key = open('private_key.pem', 'r').read()
            return rsa.sign(
                message,
                rsa.PrivateKey.load_pkcs1(private_key.encode('utf8')),
                'SHA-1')  # CloudFront requires SHA-1 hash
        cf_signer = CloudFrontSigner(key_id, rsa_signer)

    To sign with a canned policy::

        signed_url = cf_signer.generate_signed_url(
            url, date_less_than=datetime(2015, 12, 1))

    To sign with a custom policy::

        signed_url = cf_signer.generate_signed_url(url, policy=my_policy)
    """

    def __init__(self, key_id, rsa_signer):
        """Create a CloudFrontSigner.

        :type key_id: str
        :param key_id: The CloudFront Key Pair ID

        :type rsa_signer: callable
        :param rsa_signer: An RSA signer.
               Its only input parameter will be the message to be signed,
               and its output will be the signed content as a binary string.
               The hash algorithm needed by CloudFront is SHA-1.
        """
        self.key_id = key_id
        self.rsa_signer = rsa_signer

    def generate_presigned_url(self, url, date_less_than=None, policy=None):
        """Creates a signed CloudFront URL based on given parameters.

        :type url: str
        :param url: The URL of the protected object

        :type date_less_than: datetime
        :param date_less_than: The URL will expire after that date and time

        :type policy: str
        :param policy: The custom policy, possibly built by self.build_policy()

        :rtype: str
        :return: The signed URL.
        """
        both_args_supplied = date_less_than is not None and policy is not None
        neither_arg_supplied = date_less_than is None and policy is None
        if both_args_supplied or neither_arg_supplied:
            e = 'Need to provide either date_less_than or policy, but not both'
            raise ValueError(e)
        if date_less_than is not None:
            policy = self.build_policy(url, date_less_than)
        if isinstance(policy, str):
            policy = policy.encode('utf8')
        if date_less_than is not None:
            params = ['Expires=%s' % int(datetime2timestamp(date_less_than))]
        else:
            params = ['Policy=%s' % self._url_b64encode(policy).decode('utf8')]
        signature = self.rsa_signer(policy)
        params.extend([f'Signature={self._url_b64encode(signature).decode('utf8')}', f'Key-Pair-Id={self.key_id}'])
        return self._build_url(url, params)

    def _build_url(self, base_url, extra_params):
        separator = '&' if '?' in base_url else '?'
        return base_url + separator + '&'.join(extra_params)

    def build_policy(self, resource, date_less_than, date_greater_than=None, ip_address=None):
        """A helper to build policy.

        :type resource: str
        :param resource: The URL or the stream filename of the protected object

        :type date_less_than: datetime
        :param date_less_than: The URL will expire after the time has passed

        :type date_greater_than: datetime
        :param date_greater_than: The URL will not be valid until this time

        :type ip_address: str
        :param ip_address: Use 'x.x.x.x' for an IP, or 'x.x.x.x/x' for a subnet

        :rtype: str
        :return: The policy in a compact string.
        """
        moment = int(datetime2timestamp(date_less_than))
        condition = OrderedDict({'DateLessThan': {'AWS:EpochTime': moment}})
        if ip_address:
            if '/' not in ip_address:
                ip_address += '/32'
            condition['IpAddress'] = {'AWS:SourceIp': ip_address}
        if date_greater_than:
            moment = int(datetime2timestamp(date_greater_than))
            condition['DateGreaterThan'] = {'AWS:EpochTime': moment}
        ordered_payload = [('Resource', resource), ('Condition', condition)]
        custom_policy = {'Statement': [OrderedDict(ordered_payload)]}
        return json.dumps(custom_policy, separators=(',', ':'))

    def _url_b64encode(self, data):
        return base64.b64encode(data).replace(b'+', b'-').replace(b'=', b'_').replace(b'/', b'~')