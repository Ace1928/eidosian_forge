import datetime
import os
import re
from oslo_serialization import jsonutils as json
from blazarclient import exception
from blazarclient.i18n import _
def _find_resource_id_by_name(client, resource_type, name, name_key):
    resource_manager = getattr(client, resource_type)
    resources = resource_manager.list()
    named_resources = []
    key = name_key if name_key else 'name'
    for resource in resources:
        if resource[key] == name:
            named_resources.append(resource['id'])
    if len(named_resources) > 1:
        raise exception.NoUniqueMatch(message="There are more than one appropriate resources for the name '%s' and type '%s'" % (name, resource_type))
    elif named_resources:
        return named_resources[0]
    else:
        message = "Unable to find resource with name '%s'" % name
        raise exception.BlazarClientException(message=message, status_code=404)