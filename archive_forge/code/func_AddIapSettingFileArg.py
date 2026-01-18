from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def AddIapSettingFileArg(parser):
    """Add flags for the IAP setting file.

  Args:
    parser: An argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
  """
    parser.add_argument('setting_file', help='JSON or YAML file containing the IAP resource settings.\n\n       JSON example:\n         {\n           "access_settings" : {\n             "oauth_settings" : {\n                "login_hint" : {\n                   "value": "test_hint"\n                }\n             },\n             "gcip_settings" : {\n                "tenant_ids": ["tenant1-p9puj", "tenant2-y8rxc"],\n                "login_page_uri" : {\n                   "value" : "https://test.com/?apiKey=abcd_efgh"\n                }\n             },\n             "cors_settings": {\n                "allow_http_options" : {\n                   "value": true\n                }\n             }\n          },\n          "application_settings" : {\n             "csm_settings" : {\n               "rctoken_aud" : {\n                  "value" : "test_aud"\n               }\n             }\n          }\n        }\n\n       YAML example:\n       accessSettings :\n          oauthSettings:\n            loginHint: test_hint\n          gcipSettings:\n            tenantIds:\n            - tenant1-p9puj\n            - tenant2-y8rxc\n            loginPageUri: https://test.com/?apiKey=abcd_efgh\n          corsSettings:\n            allowHttpOptions: true\n       applicationSettings:\n          csmSettings:\n            rctokenAud: test_aud')