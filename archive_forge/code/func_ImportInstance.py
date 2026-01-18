from googlecloudsdk.api_lib.looker import utils
def ImportInstance(instance_ref, args, release_track):
    """Imports a Looker Instance."""
    messages_module = utils.GetMessagesModule(release_track)
    service = GetService(release_track)
    import_instance_request = messages_module.ImportInstanceRequest(gcsUri=args.source_gcs_uri)
    return service.Import(messages_module.LookerProjectsLocationsInstancesImportRequest(name=instance_ref.RelativeName(), importInstanceRequest=import_instance_request))