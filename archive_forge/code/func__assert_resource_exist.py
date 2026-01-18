import copy
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.limit.models import base
def _assert_resource_exist(self, unified_limit, target):
    try:
        service_id = unified_limit.get('service_id')
        if service_id is not None:
            PROVIDERS.catalog_api.get_service(service_id)
        region_id = unified_limit.get('region_id')
        if region_id is not None:
            PROVIDERS.catalog_api.get_region(region_id)
        project_id = unified_limit.get('project_id')
        if project_id is not None:
            project = PROVIDERS.resource_api.get_project(project_id)
            if project['is_domain']:
                unified_limit['domain_id'] = unified_limit.pop('project_id')
        domain_id = unified_limit.get('domain_id')
        if domain_id is not None:
            PROVIDERS.resource_api.get_domain(domain_id)
    except exception.ServiceNotFound:
        raise exception.ValidationError(attribute='service_id', target=target)
    except exception.RegionNotFound:
        raise exception.ValidationError(attribute='region_id', target=target)
    except exception.ProjectNotFound:
        raise exception.ValidationError(attribute='project_id', target=target)
    except exception.DomainNotFound:
        raise exception.ValidationError(attribute='domain_id', target=target)