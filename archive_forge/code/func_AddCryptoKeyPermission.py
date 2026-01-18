from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import iam as kms_iam
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.iam import util as iam_api
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import resources
def AddCryptoKeyPermission(kms_key, service_account):
    """Adds Encrypter/Decrypter role to the given service account."""
    crypto_key_ref = resources.REGISTRY.ParseRelativeName(relative_name=kms_key, collection=CRYPTO_KEY_COLLECTION)
    return kms_iam.AddPolicyBindingToCryptoKey(crypto_key_ref, service_account, 'roles/cloudkms.cryptoKeyEncrypterDecrypter')