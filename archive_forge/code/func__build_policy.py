import boto
import boto.jsonresponse
from boto.compat import json, six
from boto.resultset import ResultSet
from boto.iam.summarymap import SummaryMap
from boto.connection import AWSQueryConnection
def _build_policy(self, assume_role_policy_document=None):
    if assume_role_policy_document is not None:
        if isinstance(assume_role_policy_document, six.string_types):
            return assume_role_policy_document
    else:
        for tld, policy in DEFAULT_POLICY_DOCUMENTS.items():
            if tld is 'default':
                continue
            if self.host and self.host.endswith(tld):
                assume_role_policy_document = policy
                break
        if not assume_role_policy_document:
            assume_role_policy_document = DEFAULT_POLICY_DOCUMENTS['default']
    return json.dumps(assume_role_policy_document)