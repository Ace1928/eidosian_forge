import datetime
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth import metrics
from google.auth.compute_engine import _metadata
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import _client
def _metric_header_for_usage(self):
    return metrics.CRED_TYPE_SA_MDS