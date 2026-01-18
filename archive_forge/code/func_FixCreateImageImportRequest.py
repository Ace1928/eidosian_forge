from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
def FixCreateImageImportRequest(ref, args, req):
    """Fixes the Create Image Import request."""
    if not (args.generalize or args.license_type):
        req.imageImport.diskImageTargetDefaults.osAdaptationParameters = None
    if not args.image_name:
        req.imageImport.diskImageTargetDefaults.imageName = ref.Name()
    if args.kms_key:
        req.imageImport.diskImageTargetDefaults.encryption = GetEncryptionTransform(req.imageImport.diskImageTargetDefaults.encryption)
        req.imageImport.diskImageTargetDefaults.encryption.kmsKey = args.kms_key
        req.imageImport.encryption = GetEncryptionTransform(req.imageImport.encryption)
        req.imageImport.encryption.kmsKey = args.kms_key
    if not args.target_project:
        target = args.project or properties.VALUES.core.project.Get(required=True)
        req.imageImport.diskImageTargetDefaults.targetProject = ref.Parent().Parent().RelativeName() + '/locations/global/targetProjects/' + target
    elif '/' not in args.target_project:
        req.imageImport.diskImageTargetDefaults.targetProject = ref.Parent().Parent().RelativeName() + '/locations/global/targetProjects/' + args.target_project
    return req