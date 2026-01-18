from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddControlPlaneKMSKey(parser):
    parser.add_argument('--control-plane-kms-key', help='\n      Google Cloud KMS key that will be used to secure persistent disks of the\n      control plane VMs of a remote control plane cluster. The Edge Container\n      service account for this project must have\n      `roles/cloudkms.cryptoKeyEncrypterDecrypter` on the key.\n\n      If not provided, a Google-managed key will be used by default.\n      ')