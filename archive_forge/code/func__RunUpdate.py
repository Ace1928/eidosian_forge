from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.privateca import base as privateca_base
from googlecloudsdk.api_lib.privateca import request_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.privateca import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def _RunUpdate(self, client, messages, original_cert, args):
    labels_diff = labels_util.GetAndValidateOpsFromArgs(args)
    labels_update = labels_diff.Apply(messages.Certificate.LabelsValue, original_cert.labels)
    if not labels_update.needs_update:
        raise exceptions.InvalidArgumentException('labels', self.NO_CHANGES_MESSAGE.format(certificate=original_cert.name))
    original_cert.labels = labels_update.labels
    return client.projects_locations_caPools_certificates.Patch(messages.PrivatecaProjectsLocationsCaPoolsCertificatesPatchRequest(name=original_cert.name, certificate=original_cert, updateMask='labels', requestId=request_utils.GenerateRequestId()))