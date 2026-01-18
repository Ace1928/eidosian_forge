from __future__ import absolute_import
import io
import json
import os
import sys
import time
import webbrowser
from gcs_oauth2_boto_plugin import oauth2_client
import oauth2client.client
from six.moves import input  # pylint: disable=redefined-builtin
def OAuth2ApprovalFlow(client, scopes, launch_browser=False):
    """Run the OAuth2 flow to fetch a refresh token. Returns the refresh token."""
    flow = oauth2client.client.OAuth2WebServerFlow(client.client_id, client.client_secret, scopes, auth_uri=client.auth_uri, token_uri=client.token_uri, redirect_uri=OOB_REDIRECT_URI)
    approval_url = flow.step1_get_authorize_url()
    if launch_browser:
        sys.stdout.write('Attempting to launch a browser with the OAuth2 approval dialog at URL: %s\n\n[Note: due to a Python bug, you may see a spurious error message "object is not\ncallable [...] in [...] Popen.__del__" which can be ignored.]\n\n' % approval_url)
    else:
        sys.stdout.write('Please navigate your browser to the following URL:\n%s\n' % approval_url)
    sys.stdout.write('In your browser you should see a page that requests you to authorize access to Google Cloud Platform APIs and Services on your behalf. After you approve, an authorization code will be displayed.\n\n')
    if launch_browser and (not webbrowser.open(approval_url, new=1, autoraise=True)):
        sys.stdout.write('Launching browser appears to have failed; please navigate a browser to the following URL:\n%s\n' % approval_url)
    time.sleep(2)
    code = input('Enter the authorization code: ')
    credentials = flow.step2_exchange(code, http=client.CreateHttpRequest())
    return credentials.refresh_token