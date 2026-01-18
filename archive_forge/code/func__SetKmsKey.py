from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.secrets import api as secrets_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.secrets import args as secrets_args
from googlecloudsdk.command_lib.secrets import exceptions
from googlecloudsdk.command_lib.secrets import log as secrets_log
def _SetKmsKey(self, secret_ref, secret, kms_key, location):
    api_version = secrets_api.GetApiFromTrack(self.ReleaseTrack())
    if secret.replication.automatic:
        if location:
            raise calliope_exceptions.BadArgumentException('location', self.LOCATION_AND_AUTOMATIC_MESSAGE)
        updated_secret = secrets_api.Secrets(api_version=api_version).SetReplication(secret_ref, 'automatic', [], [kms_key])
        secrets_log.Secrets().UpdatedReplication(secret_ref)
        return updated_secret
    if secret.replication.userManaged and secret.replication.userManaged.replicas:
        if not location:
            raise calliope_exceptions.RequiredArgumentException('location', self.LOCATION_REQUIRED_MESSAGE)
        locations = []
        keys = []
        found_location = False
        for replica in secret.replication.userManaged.replicas:
            if not replica.location:
                raise exceptions.MisconfiguredReplicationError(self.MISCONFIGURED_REPLICATION_MESSAGE)
            locations.append(replica.location)
            if location == replica.location:
                found_location = True
                keys.append(kms_key)
            elif replica.customerManagedEncryption and replica.customerManagedEncryption.kmsKeyName:
                keys.append(replica.customerManagedEncryption.kmsKeyName)
        if not found_location:
            raise calliope_exceptions.InvalidArgumentException('location', self.LOCATION_NOT_IN_POLICY_MESSAGE)
        if len(locations) != len(keys):
            raise exceptions.MisconfiguredEncryptionError(self.PARTIALLY_CMEK_MESSAGE)
        updated_secret = secrets_api.Secrets(api_version=api_version).SetReplication(secret_ref, 'user-managed', locations, keys)
        secrets_log.Secrets().UpdatedReplication(secret_ref)
        return updated_secret
    raise exceptions.MisconfiguredReplicationError(self.MISCONFIGURED_REPLICATION_MESSAGE)