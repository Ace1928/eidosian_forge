from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def GetSupportedLocations(version='v1'):
    """Gets a list of supported Private CA locations for the current project."""
    if version != 'v1':
        raise exceptions.NotYetImplementedError('Unknown API version: {}'.format(version))
    client = base.GetClientInstance(api_version='v1')
    messages = base.GetMessagesModule(api_version='v1')
    project = properties.VALUES.core.project.GetOrFail()
    try:
        response = client.projects_locations.List(messages.PrivatecaProjectsLocationsListRequest(name='projects/{}'.format(project)))
        return [location.locationId for location in response.locations]
    except exceptions.HttpError as e:
        log.debug('ListLocations failed: %r.', e)
        log.debug('Falling back to hard-coded list.')
        return _V1Locations