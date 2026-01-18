import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@classmethod
def _get_aws_context(cls):
    ctx = {'cloud.provider': CLOUD_PROVIDER.AWS, 'cloud.platform': CLOUD_PLATFORM.AWS_EC2}
    try:
        r = cls.http.request('GET', AWS_METADATA_URL, headers={'X-aws-ec2-metadata-token': cls.aws_token})
        if r.status != 200:
            return ctx
        data = json.loads(r.data.decode('utf-8'))
        try:
            ctx['cloud.account.id'] = data['accountId']
        except Exception:
            pass
        try:
            ctx['cloud.availability_zone'] = data['availabilityZone']
        except Exception:
            pass
        try:
            ctx['cloud.region'] = data['region']
        except Exception:
            pass
        try:
            ctx['host.id'] = data['instanceId']
        except Exception:
            pass
        try:
            ctx['host.type'] = data['instanceType']
        except Exception:
            pass
    except Exception:
        pass
    return ctx