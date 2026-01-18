import json
import urllib3
from sentry_sdk.integrations import Integration
from sentry_sdk.api import set_context
from sentry_sdk.utils import logger
from sentry_sdk._types import TYPE_CHECKING
@classmethod
def _is_gcp(cls):
    try:
        r = cls.http.request('GET', GCP_METADATA_URL, headers={'Metadata-Flavor': 'Google'})
        if r.status != 200:
            return False
        cls.gcp_metadata = json.loads(r.data.decode('utf-8'))
        return True
    except Exception:
        return False