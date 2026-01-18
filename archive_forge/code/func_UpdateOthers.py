from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.kms import exceptions as kms_exceptions
from googlecloudsdk.command_lib.kms import flags
from googlecloudsdk.command_lib.kms import maps
from googlecloudsdk.command_lib.kms import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
def UpdateOthers(self, args, crypto_key, fields_to_update):
    client = cloudkms_base.GetClientInstance()
    messages = cloudkms_base.GetMessagesModule()
    crypto_key_ref = args.CONCEPTS.key.Parse()
    labels_update = labels_util.Diff.FromUpdateArgs(args).Apply(messages.CryptoKey.LabelsValue, crypto_key.labels)
    if labels_update.needs_update:
        new_labels = labels_update.labels
    else:
        new_labels = crypto_key.labels
    req = messages.CloudkmsProjectsLocationsKeyRingsCryptoKeysPatchRequest(name=crypto_key_ref.RelativeName(), cryptoKey=messages.CryptoKey(labels=new_labels))
    req.updateMask = ','.join(fields_to_update)
    flags.SetNextRotationTime(args, req.cryptoKey)
    flags.SetRotationPeriod(args, req.cryptoKey)
    if args.default_algorithm:
        valid_algorithms = maps.VALID_ALGORITHMS_MAP[crypto_key.purpose]
        if args.default_algorithm not in valid_algorithms:
            raise kms_exceptions.UpdateError('Update failed: Algorithm {algorithm} is not valid. Here are the valid algorithm(s) for purpose {purpose}: {all_algorithms}'.format(algorithm=args.default_algorithm, purpose=crypto_key.purpose, all_algorithms=', '.join(valid_algorithms)))
        req.cryptoKey.versionTemplate = messages.CryptoKeyVersionTemplate(algorithm=maps.ALGORITHM_MAPPER.GetEnumForChoice(args.default_algorithm))
    try:
        response = client.projects_locations_keyRings_cryptoKeys.Patch(req)
    except apitools_exceptions.HttpError:
        return None
    return response