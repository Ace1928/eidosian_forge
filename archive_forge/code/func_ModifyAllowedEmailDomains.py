from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.console import console_io
def ModifyAllowedEmailDomains(unused_instance_ref, args, patch_request):
    """Python hook to modify allowed email domains in looker instance update request."""
    if args.IsSpecified('allowed_email_domains'):
        _WarnForAdminSettingsUpdate()
        patch_request.instance.adminSettings.allowedEmailDomains = args.allowed_email_domains
        patch_request = AddFieldToUpdateMask('admin_settings.allowed_email_domains', patch_request)
    return patch_request