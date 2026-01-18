from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
def ConstructServiceBindingServiceNameFromArgs(unused_ref, args, request):
    sd_service_name = 'projects/' + properties.VALUES.core.project.Get() + '/locations/' + args.service_directory_region + '/namespaces/' + args.service_directory_namespace + '/services/' + args.service_directory_service
    arg_utils.SetFieldInMessage(request, 'serviceBinding.service', sd_service_name)
    return request