import os
import pprint
from twython import Twython
def add_access_token(creds_file=None):
    """
    For OAuth 2, retrieve an access token for an app and append it to a
    credentials file.
    """
    if creds_file is None:
        path = os.path.dirname(__file__)
        creds_file = os.path.join(path, 'credentials2.txt')
    oauth2 = credsfromfile(creds_file=creds_file)
    app_key = oauth2['app_key']
    app_secret = oauth2['app_secret']
    twitter = Twython(app_key, app_secret, oauth_version=2)
    access_token = twitter.obtain_access_token()
    tok = f'access_token={access_token}\n'
    with open(creds_file, 'a') as infile:
        print(tok, file=infile)