from libcloud.backup.base import (
from libcloud.backup.types import BackupTargetType, BackupTargetJobStatusType
from libcloud.common.google import GoogleResponse, GoogleBaseConnection
from libcloud.utils.iso8601 import parse_date
class GCEConnection(GoogleBaseConnection):
    """
    Connection class for the GCE driver.

    GCEConnection extends :class:`google.GoogleBaseConnection` for 2 reasons:
      1. modify request_path for GCE URI.
      2. Implement gce_params functionality described below.

    If the parameter gce_params is set to a dict prior to calling request(),
    the URL parameters will be updated to include those key/values FOR A
    SINGLE REQUEST. If the response contains a nextPageToken,
    gce_params['pageToken'] will be set to its value. This can be used to
    implement paging in list:

    >>> params, more_results = {'maxResults': 2}, True
    >>> while more_results:
    ...     driver.connection.gce_params=params
    ...     driver.ex_list_urlmaps()
    ...     more_results = 'pageToken' in params
    ...
    [<GCEUrlMap id="..." name="cli-map">, <GCEUrlMap id="..." name="lc-map">]
    [<GCEUrlMap id="..." name="web-map">]
    """
    host = 'www.googleapis.com'
    responseCls = GCEResponse

    def __init__(self, user_id, key, secure, auth_type=None, credential_file=None, project=None, **kwargs):
        super().__init__(user_id, key, secure=secure, auth_type=auth_type, credential_file=credential_file, **kwargs)
        self.request_path = '/compute/{}/projects/{}'.format(API_VERSION, project)
        self.gce_params = {}

    def pre_connect_hook(self, params, headers):
        """
        Update URL parameters with values from self.gce_params.

        @inherits: :class:`GoogleBaseConnection.pre_connect_hook`
        """
        params, headers = super().pre_connect_hook(params, headers)
        if self.gce_params:
            params.update(self.gce_params)
        return (params, headers)

    def request(self, *args, **kwargs):
        """
        Perform request then do GCE-specific processing of URL params.

        @inherits: :class:`GoogleBaseConnection.request`
        """
        response = super().request(*args, **kwargs)
        if self.gce_params:
            if 'nextPageToken' in response.object:
                self.gce_params['pageToken'] = response.object['nextPageToken']
            elif 'pageToken' in self.gce_params:
                del self.gce_params['pageToken']
            self.gce_params = None
        return response