from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def AddCredentialFlag(parser):
    """Add the credential argument."""
    parser.add_argument('--credential', help='Set the default credential that Deployment Manager uses to call underlying APIs of a deployment. Use PROJECT_DEFAULT to set deployment credential same as the credential of its owning project. Use serviceAccount:email to set default credential using provided service account.', dest='credential')