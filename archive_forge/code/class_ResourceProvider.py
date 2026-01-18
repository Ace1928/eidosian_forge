from openstack import exceptions
from openstack import resource
from openstack import utils
class ResourceProvider(resource.Resource):
    resource_key = None
    resources_key = 'resource_providers'
    base_path = '/resource_providers'
    allow_create = True
    allow_fetch = True
    allow_commit = True
    allow_delete = True
    allow_list = True
    _query_mapping = resource.QueryParameters('name', 'member_of', 'resources', 'in_tree', 'required', id='uuid')
    _max_microversion = '1.20'
    aggregates = resource.Body('aggregates', type=list, list_type=str)
    id = resource.Body('uuid', alternate_id=True)
    generation = resource.Body('generation')
    links = resource.Body('links')
    name = resource.Body('name')
    parent_provider_id = resource.Body('parent_provider_uuid')
    root_provider_id = resource.Body('root_provider_uuid')

    def fetch_aggregates(self, session):
        """List aggregates set on the resource provider

        :param session: The session to use for making this request
        :return: The resource provider with aggregates populated
        """
        url = utils.urljoin(self.base_path, self.id, 'aggregates')
        microversion = self._get_microversion(session, action='fetch')
        response = session.get(url, microversion=microversion)
        exceptions.raise_from_response(response)
        data = response.json()
        updates = {'aggregates': data['aggregates']}
        if utils.supports_microversion(session, '1.19'):
            updates['generation'] = data['resource_provider_generation']
        self._body.attributes.update(updates)
        return self

    def set_aggregates(self, session, aggregates=None):
        """Replaces aggregates on the resource provider

        :param session: The session to use for making this request
        :param list aggregates: List of aggregates
        :return: The resource provider with updated aggregates populated
        """
        url = utils.urljoin(self.base_path, self.id, 'aggregates')
        microversion = self._get_microversion(session, action='commit')
        body = {'aggregates': aggregates or []}
        if utils.supports_microversion(session, '1.19'):
            body['resource_provider_generation'] = self.generation
        response = session.put(url, json=body, microversion=microversion)
        exceptions.raise_from_response(response)
        data = response.json()
        updates = {'aggregates': data['aggregates']}
        if 'resource_provider_generation' in data:
            updates['resource_provider_generation'] = data['resource_provider_generation']
        self._body.attributes.update(updates)
        return self