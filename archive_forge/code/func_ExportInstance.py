from googlecloudsdk.api_lib.looker import utils
def ExportInstance(instance_ref, args, release_track):
    """Exports a Looker Instance."""
    messages_module = utils.GetMessagesModule(release_track)
    service = GetService(release_track)
    encryption_config = messages_module.ExportEncryptionConfig(kmsKeyName=args.kms_key)
    export_instance_request = messages_module.ExportInstanceRequest(gcsUri=args.target_gcs_uri, encryptionConfig=encryption_config)
    return service.Export(messages_module.LookerProjectsLocationsInstancesExportRequest(name=instance_ref.RelativeName(), exportInstanceRequest=export_instance_request))